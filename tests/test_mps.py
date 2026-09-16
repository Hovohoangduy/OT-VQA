"""Opt-in Apple MPS smoke test: OT, decoder, loss, and SAN question LSTM."""

import os
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from model.decoder_model import Decoder
from model.optimal_transport import OTConfig, OptimalTransportFusion


@unittest.skipUnless(
    os.environ.get("OT_VQA_TEST_MPS") == "1" and torch.backends.mps.is_available(),
    "set OT_VQA_TEST_MPS=1 on a Mac with available MPS",
)
class MPSSmokeTests(unittest.TestCase):
    def test_ot_decoder_and_lstm_backward_on_apple_gpu(self):
        device = torch.device("mps")
        torch.mps.manual_seed(9)
        fusion = OptimalTransportFusion(
            6, 7, 12,
            OTConfig(ot_dim=8, epsilon=0.1, max_iterations=20),
        ).to(device)
        decoder = Decoder(12, 24, 3, 0.0, 1).to(device)
        visual = torch.randn(2, 4, 6, device=device, requires_grad=True)
        question = torch.randn(2, 5, 7, device=device, requires_grad=True)
        visual_mask = torch.tensor(
            [[False] * 4, [False, False, False, True]], device=device
        )
        question_mask = torch.tensor(
            [[False, False, False, True, True], [False] * 5], device=device
        )
        transport = fusion(
            visual, question, visual_mask, question_mask,
            return_diagnostics=True,
        )
        target = torch.randn(2, 3, 12, device=device)
        causal_mask = torch.triu(
            torch.ones(3, 3, dtype=torch.bool, device=device), diagonal=1
        )
        memory_mask = question_mask[:, None, None, :].expand(-1, 1, 3, -1)
        decoded = decoder(
            transport.fused_tokens, target, causal_mask, memory_mask
        )
        labels = torch.tensor([[1, 2, 0], [2, 1, 1]], device=device)
        logits = torch.nn.Linear(12, 3, device=device)(decoded)
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=0
        ) + transport.transport_cost.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(transport.plan).all().item())
        self.assertIsNotNone(fusion.pairwise_cost.learned[0].weight.grad)

        # SAN uses a packed LSTM question summary; cover that MPS kernel too.
        lstm = torch.nn.LSTM(7, 12, batch_first=True).to(device)
        sequence = torch.randn(2, 5, 7, device=device, requires_grad=True)
        packed = pack_padded_sequence(
            sequence, torch.tensor([5, 3]), batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = lstm(packed)
        hidden.square().mean().backward()
        self.assertTrue(torch.isfinite(hidden).all().item())


if __name__ == "__main__":
    unittest.main()
