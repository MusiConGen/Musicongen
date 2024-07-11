from audiocraft.data.audio import audio_write
import audiocraft.models
from glob import glob
import time
import datetime
import multiprocessing

import librosa
import numpy as np
import pandas as pd
import os
import gc

import json
from pathlib import Path
import torch
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from evaluation.btc_model.btc_model import *
from evaluation.btc_model.utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from evaluation import mir_eval

from random import choice
from audiocraft.data.btc_chords import Chords
CHORDS = Chords()


def get_chroma_chord(chord_path, n_frames_feat):
  intervals = []
  labels = []
  feat_chord = np.zeros((12, n_frames_feat)) # root| ivs
  with open(chord_path, 'r') as f:
      for line in f.readlines():
          splits = line.split()
          if len(splits) == 3:
              st_sec, ed_sec, ctag = splits
              st_sec = float(st_sec)
              ed_sec = float(ed_sec)

              st_frame = int(st_sec*feat_hz)
              ed_frame = int(ed_sec*feat_hz)

              mhot = CHORDS.chord(ctag)
              final_vec = np.roll(mhot[2], mhot[0])

              final_vec = final_vec[..., None] # [B, T]
              feat_chord[:, st_frame:ed_frame] = final_vec
  feat_chord = torch.from_numpy(feat_chord)
  return feat_chord

def get_bpm_beat(beat_path, bpm):
  beats_np = np.load(beat_path, allow_pickle=True)
  feat_beats = np.zeros((2, n_frames_feat))
  meter = int(max(beats_np.T[1]))

  beat_time_gap = 60 / bpm
  beat_gap = 60 / bpm * feat_hz
  
  beat_time = np.arange(0, duration, beat_time_gap)
  beat_frame = np.round(np.arange(0, n_frames_feat, beat_gap)).astype(int)
  if beat_frame[-1] == n_frames_feat:
    beat_frame = beat_frame[:-1]
  bar_frame = beat_frame[::meter]
  
  feat_beats[0, beat_frame] = 1
  feat_beats[1, bar_frame] = 1
  kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
  feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
  beat_events = feat_beats[0] + feat_beats[1]
  beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]
  return beat_events, beat_time, meter

def chord_model_init():
  # device = torch.device("cpu")
  device = torch.device("cuda")
  config = HParams.load("./evaluation/btc_model/run_config.yaml")
  config.feature['large_voca'] = True
  config.model['num_chords'] = 170
  model_file = './evaluation/btc_model/test/btc_model_large_voca.pt'
  idx_to_chord = idx2voca_chord()
  model = BTC_model(config=config.model).to(device)
  if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
  return model, config, device, mean, std, idx_to_chord


def extract_chords(audio_pth):
  chord_dir = os.path.dirname(audio_pth).replace("audio", "chord")
  os.makedirs(chord_dir, exist_ok=True)
  chord_path = os.path.join(chord_dir, os.path.basename(audio_pth.replace('wav', 'lab')))

  feature, feature_per_second, song_length_second = audio_file_to_features(audio_pth, config)
  feature = feature.T
  feature = (feature - mean) / std
  time_unit = feature_per_second
  n_timestep = config.model['timestep']
  num_pad = n_timestep - (feature.shape[0] % n_timestep)
  feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
  num_instance = feature.shape[0] // n_timestep

  start_time = 0.0
  lines = []
  with torch.no_grad():
    chord_model.eval()
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    for t in range(num_instance):
      self_attn_output, _ = chord_model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
      prediction, _ = chord_model.output_layer(self_attn_output)
      prediction = prediction.squeeze()
      for i in range(n_timestep):
        if t == 0 and i == 0:
          prev_chord = prediction[i].item()
          continue
        if prediction[i].item() != prev_chord:
          lines.append(
              '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
          start_time = time_unit * (n_timestep * t + i)
          prev_chord = prediction[i].item()
        if t == num_instance - 1 and i + num_pad == n_timestep:
          if start_time != time_unit * (n_timestep * t + i):
            lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
          break
  
  # lab file write
  with open(chord_path, 'w') as f:
    for line in lines:
      f.write(line)

  return True


def extract_beats(audio_pth):
  beat_dir = os.path.dirname(audio_pth).replace("audio", "beat")
  os.makedirs(beat_dir, exist_ok=True)
  beat_path = os.path.join(beat_dir, os.path.basename(audio_pth.replace('wav', 'npy')))
  
  est_beat = proc(act(audio_pth)).T[0]
  np.save(beat_path, est_beat)

  return True


def measure_chords(chord_path, ref_chord_pth):
  
  (est_interval, est_label) = mir_eval.io.load_labeled_intervals(chord_path)
  (ref_interval, ref_label) = mir_eval.io.load_labeled_intervals(ref_chord_pth)
  est_interval, est_label = mir_eval.util.adjust_intervals(est_interval, est_label, ref_interval.min(),ref_interval.max(), 
                                                               mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
  (interval, ref_label, est_label) = mir_eval.util.merge_labeled_intervals(ref_interval, ref_label, est_interval, est_label)
  dur = mir_eval.util.intervals_to_durations(interval)

  # compute majmin score
  comparisons = mir_eval.chord.majmin(ref_label, est_label)
  majmin_score = mir_eval.chord.weighted_accuracy(comparisons, dur)

  # compute triads score
  comparisons = mir_eval.chord.triads(ref_label, est_label)
  triads_score = mir_eval.chord.weighted_accuracy(comparisons, dur)

  # compute tetrads score
  comparisons = mir_eval.chord.tetrads(ref_label, est_label)
  tetrads_score = mir_eval.chord.weighted_accuracy(comparisons, dur)

  return majmin_score, triads_score, tetrads_score


def measure_bpm_beats(beat_path, ref_beat):
  
  est_beat = np.load(beat_path)
  # ref_beat = np.load(ref_beat_path)
  f_measure = mir_eval.beat.f_measure(ref_beat, est_beat, f_measure_threshold=0.07)
  return f_measure


if __name__ == "__main__":
  # set hparams
  output_dir = 'musicongen' ### change this directory
  ckpt_pth = './ckpt/musicongen'
  top_k = 250
  duration = 30
  bs = 10


  output_dir = output_dir+ '_rwc'
  print(output_dir)

  df = pd.DataFrame(columns=["audio_path", "bpm", "meter", "desc", "ref_dir", "f_measure", "majmin", "triads", "tetrads"])

  # get musdb data
  rwc_file_lst = glob('dataset/rwc_pop/clip/*/*/no_vocal.wav')
  rwc_file_lst.sort()
  num_samples = len(rwc_file_lst)
  print(f"Total evaluation data number: {num_samples}")


  # lab chord to chromagram
  n_frames_feat = 32000 * duration // 640
  feat_hz = 32000/640

  beats, chords, descs, bpms, meters, ref_dirs, ref_beat_times = [], [], [], [], [], [], []
  
  desc_options = ["A reflective solo piano piece with intricate melodies, rich harmonies, and a deep emotional resonance, showcasing the instrument's versatility. Instrument: piano.",
                "A smooth acid jazz track with a laid-back groove, silky electric piano, and a cool bass, providing a modern take on jazz. Instruments: electric piano, bass, drums.",
                "A soothing classical guitar piece with intricate fingerpicking patterns, warm melodies, and a romantic flair, ideal for a peaceful evening. Instrument: classical guitar.",
                "A gritty blues rock track with wailing electric guitar solos, a solid rhythm section, and a raw, emotional edge. Instruments: electric guitar, bass, drums.",
                "A high-energy funk tune with slap bass, rhythmic guitar riffs, and a tight horn section, guaranteed to get you grooving. Instruments: bass, guitar, trumpet, saxophone, drums.",
                "An epic metal anthem with fast-paced riffs, thunderous drums, and a soaring guitar solo, capturing the power and intensity of the genre. Instruments: electric guitar, bass guitar, drums.",
                "A sultry jazz ballad with a soulful saxophone lead, gentle piano accompaniment, and a smooth rhythm section, evoking a sense of romance and nostalgia. Instruments: saxophone, piano, bass, drums.",
                "A classic rock n' roll tune with catchy guitar riffs, driving drums, and a pulsating bass line, reminiscent of the golden era of rock. Instruments: electric guitar, bass, drums.",
                "An upbeat funk track with funky rhythm guitar, a bouncy bass groove, and vibrant brass stabs, creating a danceable beat. Instruments: guitar, bass, trumpet, saxophone, drums.",
                "A heavy metal onslaught with double kick drum madness, aggressive guitar riffs, and an unrelenting bass, embodying the spirit of metal. Instruments: electric guitar, bass guitar, drums.",
                "A smooth jazz fusion track with intricate keyboard melodies, a complex rhythm section, and a fusion of electronic and acoustic sounds. Instruments: keyboard, electric guitar, bass, drums.",
                "A laid-back blues shuffle with a relaxed tempo, warm guitar tones, and a comfortable groove, perfect for a slow dance or a night in. Instruments: electric guitar, bass, drums.",
                "A psychedelic rock experience with swirling guitar effects, deep bass vibrations, and expansive drum patterns, transporting the listener to another era. Instruments: electric guitar, bass, drums.",
                "A melancholic blues ballad with soul-stirring guitar solos, a tender rhythm section, and a deep emotional resonance. Instruments: guitar, bass, drums.",
                "A vibrant jazz waltz with a lilting piano melody, swinging rhythms, and a playful interplay between the instruments, evoking an elegant ballroom dance. Instruments: piano, double bass, drums.",
                "A rockabilly romp with twangy guitars, an upright bass slap, and a driving beat, combining the best of rock n' roll and country vibes. Instruments: electric guitar, upright bass, drums."
                ]
  
  # load bpm
  bpm_json_path = "dataset/rwc_pop/bpm.json"
  with open(bpm_json_path ,'r') as f:
      json_str = f.read()
  bpm_json = json.loads(json_str)

  for i, wav_path in enumerate(rwc_file_lst):
    dir_name = os.path.dirname(wav_path)
    chord_path = os.path.join(dir_name, "chord.lab")
    beat_path = os.path.join(dir_name, "beats.npy")
    dn = dir_name.split('/')[-2] # DiscX_00X

    # get beat feature and bpm
    bpm = bpm_json[dn]
    feat_beats, beat_time, meter = get_bpm_beat(beat_path, bpm)

    # desc
    desc = desc_options[i % len(desc_options)]
    descs.append(desc)

    # get chord
    chord_feats = get_chroma_chord(chord_path, n_frames_feat)

    beats.append(feat_beats)
    bpms.append(bpm)
    meters.append(meter)
    chords.append(chord_feats)

    ref_dirs.append(dir_name)
    ref_beat_times.append(beat_time)


  # generate wavs
  
  if not os.path.exists(os.path.join('./evaluation/eval_data', output_dir, 'audio')):

    # load your finetune
    musicgen = audiocraft.models.MusicGen.get_pretrained(ckpt_pth)
    musicgen.set_generation_params(duration=duration, extend_stride=duration//2, top_k = top_k) # tuning top k

    print(f"start generating music...")
    wav = []
    print(f"total {int(np.ceil(num_samples/bs))} batches...")
    for i in range(int(np.ceil(num_samples/bs))):
      print(f"starting {i+1} batch...")
      start_time = time.time()
      
      temp = musicgen.generate_for_eval(descs[i*bs:(i+1)*bs], 
                                        chords[i*bs:(i+1)*bs],
                                        beats[i*bs:(i+1)*bs], 
                                        bpms[i*bs:(i+1)*bs]
                                        )
      
      wav.extend(temp.cpu())
      end_time = time.time()
      runtime = end_time - start_time
      print('batch time:', str(datetime.timedelta(seconds=runtime))+'\n')
  
    # save and display generated audio
    for idx, one_wav in enumerate(wav):
      sav_path = os.path.join('./evaluation/eval_data', output_dir, 'audio', f'musdb_{idx}').replace(" ", "_")
      audio_write(sav_path, one_wav.cpu(), musicgen.sample_rate, strategy='loudness', loudness_compressor=True)
  
    # delete model for saving memory
    del musicgen
    gc.collect()
    torch.cuda.empty_cache()  


  ####### extract chord and beat ########

  # init model
  chord_model, config, device, mean, std, idx_to_chord = chord_model_init()
  proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
  act = RNNDownBeatProcessor()

  # extract chord and beat
  print("start extracting...")
  start_time = time.time()

  wav_pth_lst = glob(os.path.join('./evaluation/eval_data', output_dir, 'audio', '*.wav'))
  beat_pairs = []
  chord_pairs = []
  assert len(wav_pth_lst) == len(ref_dirs), f"the lengths of wav_pth_lst: {len(wav_pth_lst)} and ref_dirs: {len(ref_dirs)} are mismatch"

  for i in range(len(wav_pth_lst)):
    beat_pairs.append(wav_pth_lst[i])
    chord_pairs.append(wav_pth_lst[i])

  # extract beat with multi-processing
  cpu_count = 16
  with multiprocessing.Pool(processes=cpu_count) as pool:
    results = []
    if not os.path.exists(os.path.join('./evaluation/eval_data', output_dir, 'beat')):
      for result in pool.map(extract_beats, beat_pairs):
        results.append(result)

  # extract chord  
  for wav_pth in chord_pairs:
    if not os.path.exists(os.path.join('./evaluation/eval_data', output_dir, 'chord', os.path.basename(wav_pth).replace(".wav", ".lab"))):
      extract_chords(wav_pth)
  
  end_time = time.time()
  runtime = end_time - start_time
  print('extracting time:', str(datetime.timedelta(seconds=runtime))+'\n')


  # measure chord and beat
  print("start measuring...")
  start_time = time.time()

  for idx, audio_pth in enumerate(wav_pth_lst):
    
    chord_dir = os.path.dirname(audio_pth).replace("audio", "chord")
    chord_path = os.path.join(chord_dir, os.path.basename(audio_pth.replace('wav', 'lab')))
    ref_chord_pth = os.path.join(ref_dirs[idx], 'chord.lab')

    beat_dir = os.path.dirname(audio_pth).replace("audio", "beat")
    beat_path = os.path.join(beat_dir, os.path.basename(audio_pth.replace('wav', 'npy')))
    ref_beat_path = os.path.join(ref_dirs[idx], 'beats.npy')
    ref_beat_time = ref_beat_times[idx]

    maj_min, triads, tetrads = measure_chords(chord_path, ref_chord_pth)
    f_m = measure_bpm_beats(beat_path, ref_beat_time)
    
    df.loc[len(df)] = [os.path.abspath(audio_pth), bpms[idx], meters[idx], descs[idx], ref_dirs[idx], f_m, maj_min, triads, tetrads]
  
  df.to_csv(os.path.join('./evaluation/eval_data', output_dir, 'meta.csv'), index=False)

  end_time = time.time()
  runtime = end_time - start_time
  print('measuring time:', str(datetime.timedelta(seconds=runtime))+'\n')