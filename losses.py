import torch
import torchaudio
from transformers import AutoModel


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    r_losses, g_losses はログ記録 (TensorBoard) 専用に使われる値であり、
    通常ステップ (log_this_step=False) では捨てられる。
    そのため、ここでは常に .item() を呼ばずテンソルのまま返す。
    .item() は GPU->CPU 同期を強制するため、これを毎ステップ呼ぶと
    ログを取らないステップでも無駄な同期コストを払うことになり、
    特に1ステップの計算時間が短い/同期レイテンシに敏感な環境
    (例: 一部のクラウド上のH100/B200/B300など) で顕著なオーバーヘッドになりうる。
    値が実際に必要な呼び出し側 (ログ記録時のみ) で .item() すること。
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss)  # .item() を呼ばずテンソルのまま格納
        g_losses.append(g_loss)  # .item() を呼ばずテンソルのまま格納

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def can_share_embeddings(self):
        """Sharing is equivalent only for a frozen, deterministic extractor."""
        return not self.wavlm.training and not any(
            param.requires_grad for param in self.wavlm.parameters()
        )

    def _hidden_states(self, wav):
        return self.wavlm(
            input_values=self.resample(wav), output_hidden_states=True
        ).hidden_states

    @staticmethod
    def _discriminator_input(embeddings):
        return (
            torch.stack(embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )

    def generator_losses(self, wav, y_rec, *, wav_embeddings=None):
        """Compute both G losses from one differentiable WavLM forward.

        The optional real embeddings belong to this batch's D pass. Never cache
        the generated embeddings across D/G: G needs a fresh autograd graph.
        """
        if not self.can_share_embeddings():
            return self(wav, y_rec), self.generator(y_rec)
        if wav_embeddings is None:
            with torch.no_grad():
                wav_embeddings = self._hidden_states(wav)
        y_rec_embeddings = self._hidden_states(y_rec)
        feature = sum(
            torch.mean(torch.abs(er.detach() - eg))
            for er, eg in zip(wav_embeddings, y_rec_embeddings)
        )
        prediction = self.wd(self._discriminator_input(y_rec_embeddings))
        return feature.mean(), torch.mean((1 - prediction) ** 2)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec, *, return_real_embeddings=False):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        loss = loss_disc_f.mean()
        if return_real_embeddings:
            return loss, wav_embeddings
        return loss

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
