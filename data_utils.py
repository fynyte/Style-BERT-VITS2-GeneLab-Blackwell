import os
import random
import sys

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from config import get_config
from mel_processing import mel_spectrogram_torch, spectrogram_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParametersData
from style_bert_vits2.models.utils import load_filepaths_and_text, load_wav_to_torch
from style_bert_vits2.nlp import cleaned_text_to_sequence


config = get_config()
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text: str, hparams: HyperParametersData):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.use_jp_extra = getattr(hparams, "use_jp_extra", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        # __init__時に全サンプルをメモリへプリロードするかどうか。
        # HyperParametersData 側に preload_all_to_memory が無い場合は False 扱い。
        self.preload_all_to_memory = getattr(
            #hparams, "preload_all_to_memory", False
            hparams, "preload_all_to_memory", True
        )
        self.cache: list = []

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

        if self.preload_all_to_memory:
            self._preload()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        text_lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True
        ):
            audiopath = f"{_id}"
            # if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
            phones = phones.split(" ")
            tone = [int(i) for i in tone.split(" ")]
            word2ph = [int(i) for i in word2ph.split(" ")]
            audiopaths_sid_text_new.append(
                [audiopath, spk, language, text, phones, tone, word2ph]
            )
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            # get_text() と同じ規則 (add_blank なら intersperse で 2n+1) でテキスト長を記録。
            # 静的形状 collate (StaticShapeTable) がバケットごとのテキスト長の上限を決めるのに使う。
            text_lengths.append(len(phones) * 2 + 1 if self.add_blank else len(phones))
            # else:
            #     skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.text_lengths = text_lengths

    def _preload(self):
        """
        __init__時に全サンプル（音声・スペクトログラム・BERT特徴量・スタイルベクトル）を
        あらかじめデシリアライズしてメモリ上のリストに保持しておく。
        これにより __getitem__ 時のファイルI/O・torch.load/np.loadのデコードコストがなくなる。
        """
        logger.info("Preloading all samples into memory (this may take a while)...")
        self.cache = [None] * len(self.audiopaths_sid_text)
        failed = 0
        for i in tqdm(
            range(len(self.audiopaths_sid_text)), file=sys.stdout, dynamic_ncols=True
        ):
            try:
                self.cache[i] = self._load_sample(self.audiopaths_sid_text[i])
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to preload sample {i}: {e}")
                self.cache[i] = None
        logger.info(
            f"Preload finished: {len(self.cache) - failed}/{len(self.cache)} samples "
            f"loaded into memory ({failed} failed, will fall back to on-the-fly loading)."
        )

    def _load_sample(self, audiopath_sid_text):
        """
        1サンプル分を実際にディスク（または /dev/shm 等）から読み込み、
        get_audio_text_speaker_pair が返すのと同じ形式のタプルを構築する。
        通常の都度読み込みとプリロード（_preload）の両方から呼ばれる共通処理。
        """
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, en_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        if self.use_jp_extra:
            return (phones, spec, wav, sid, tone, language, ja_bert, style_vec)
        else:
            return (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )

    def get_audio_text_speaker_pair(self, audiopath_sid_text, index=None):
        # プリロード済みキャッシュがあればそれを返す（ファイルI/O・デコードを省略）
        if self.preload_all_to_memory and index is not None:
            cached = self.cache[index]
            if cached is not None:
                return cached
            # プリロードに失敗していたサンプルはここで都度読み込みにフォールバック
        return self._load_sample(audiopath_sid_text)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert_ori = torch.load(bert_path)
            assert bert_ori.shape[-1] == len(phone)
        except Exception as e:
            logger.warning("Bert load Failed")
            logger.warning(e)

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "JP":
            bert = torch.zeros(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "EN":
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = bert_ori
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(
            self.audiopaths_sid_text[index], index=index
        )

    def __len__(self):
        return len(self.audiopaths_sid_text)


class StaticShapeTable:
    """バッチの形状をバケット単位で固定するための表 (CUDA Graph 用)。

    従来の collate はバッチ内の最大長までパディングするので、バッチごとにテンソル形状が変わる。
    CUDA Graph は形状ごとに 1 つ必要なので、ここではスペクトログラム長をバケット上限
    (DistributedBucketSampler の境界) まで、テキスト長をそのバケットで起こりうる最大値まで
    パディングして、形状の種類をバケット数 (高々 8 前後) に抑える。
    パディングはゼロ埋め + x_lengths / spec_lengths によるマスクなので、学習の意味は変わらない
    (MAS ノイズの std だけは models_jp_extra._std_over_batch_max で従来と同じ範囲に揃えている)。
    """

    def __init__(
        self,
        spec_bounds,
        text_pad,
        hop_length,
    ):
        self.spec_bounds = sorted(int(b) for b in spec_bounds)
        self.text_pad = {int(k): int(v) for k, v in text_pad.items()}
        self.hop_length = int(hop_length)
        self.n_dynamic = 0  # テーブルに収まらず従来どおりの動的形状になった回数

    @classmethod
    def from_sampler(cls, dataset, sampler, hop_length, text_multiple=8):
        """DistributedBucketSampler の境界 (空バケット除去後) とデータセットの長さから作る。"""
        bounds = list(sampler.boundaries[1:])
        lengths = dataset.lengths
        text_lengths = dataset.text_lengths
        text_pad = {}
        for b in bounds:
            # サンプラーの長さ (ファイルサイズ由来) は実際のフレーム数より高々 1 大きいので、
            # 「b + 1 以下のサンプル」を累積して余裕を持たせる (境界ちょうどのバッチも収まる)
            m = 1
            for length, tl in zip(lengths, text_lengths):
                if length <= b + 1 and tl > m:
                    m = tl
            text_pad[b] = -(-m // text_multiple) * text_multiple
        return cls(bounds, text_pad, hop_length)

    def resolve(self, text_len, spec_len, wav_len):
        """バッチ内の最大長 (text, spec, wav) を静的形状に引き上げる。表に収まらなければそのまま返す。"""
        for b in self.spec_bounds:
            if spec_len <= b:
                return (
                    max(text_len, self.text_pad[b]),
                    b,
                    max(wav_len, b * self.hop_length),
                )
        self.n_dynamic += 1
        return text_len, spec_len, wav_len

    def shapes(self):
        return [(b, self.text_pad[b]) for b in self.spec_bounds]


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False, use_jp_extra=False, static_table=None):
        self.return_ids = return_ids
        self.use_jp_extra = use_jp_extra
        # StaticShapeTable を渡すと、バッチ内最大長ではなくバケット上限までパディングする
        self.static_table = static_table

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        if self.static_table is not None:
            max_text_len, max_spec_len, max_wav_len = self.static_table.resolve(
                max_text_len, max_spec_len, max_wav_len
            )

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        # This is ZH bert if not use_jp_extra, JA bert if use_jp_extra
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        if not self.use_jp_extra:
            ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
            en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        style_vec = torch.FloatTensor(len(batch), 256)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        if not self.use_jp_extra:
            ja_bert_padded.zero_()
            en_bert_padded.zero_()
        style_vec.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            if self.use_jp_extra:
                style_vec[i, :] = row[7]
            else:
                ja_bert = row[7]
                ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

                en_bert = row[8]
                en_bert_padded[i, :, : en_bert.size(1)] = en_bert
                style_vec[i, :] = row[9]

        if self.use_jp_extra:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                style_vec,
            )
        else:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                ja_bert_padded,
                en_bert_padded,
                style_vec,
            )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        logger.info(f"Bucket info: {self.num_samples_per_bucket}")
        # logger.info(
        #     f"Unused samples: {len(self.lengths) - sum(self.num_samples_per_bucket)}"
        # )
        # ↑マイナスになることあるし、別にこれは使われないサンプル数ではないようだ……
        # バケットの仕組みはよく分からない

        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            logger.info("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
