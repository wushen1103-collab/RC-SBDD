import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np


if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool


def main():
    root = Path(__file__).resolve().parents[1] / "external" / "MolCRAFT" / "MolCRAFT"
    sys.modules.setdefault("fire", types.ModuleType("fire"))
    rdkit_six = types.ModuleType("rdkit.six")
    rdkit_six_moves = types.ModuleType("rdkit.six.moves")
    rdkit_six_moves.cPickle = pickle
    rdkit_six.iteritems = lambda mapping: mapping.items()
    rdkit_six.moves = rdkit_six_moves
    sys.modules.setdefault("rdkit.six", rdkit_six)
    sys.modules.setdefault("rdkit.six.moves", rdkit_six_moves)
    docking_vina = types.ModuleType("core.evaluation.docking_vina")

    class GenerationOnlyVinaTask:
        @classmethod
        def from_generated_mol(cls, *args, **kwargs):
            return cls()

        def run(self, *args, **kwargs):
            return [{"affinity": float("nan")}]

    docking_vina.VinaDockingTask = GenerationOnlyVinaTask
    sys.modules.setdefault("core.evaluation.docking_vina", docking_vina)
    posecheck = types.ModuleType("posecheck")

    class GenerationOnlyPoseCheck:
        def load_protein_from_pdb(self, *args, **kwargs):
            return None

        def load_ligands_from_mols(self, *args, **kwargs):
            return None

        def calculate_strain_energy(self):
            return [float("nan")]

        def calculate_clashes(self):
            return [float("nan")]

        def calculate_interactions(self):
            import pandas as pd

            return pd.DataFrame()

    posecheck.PoseCheck = GenerationOnlyPoseCheck
    sys.modules.setdefault("posecheck", posecheck)
    os.chdir(root)
    sys.path.insert(0, str(root))
    script = root / "sample_for_pocket.py"
    source = script.read_text(encoding="utf-8")
    source = source.replace(
        "        devices=1,\n",
        "        accelerator='gpu',\n        devices=1,\n",
    )
    source = source.replace(
        "    call(protein_path, ligand_path)\n",
        "    import core.callbacks.validation_callback_for_sample as _sample_cb\n"
        "    _sample_cb.OUT_DIR = os.environ.get('MOLCRAFT_OUT_DIR', './output')\n"
        "    OUT_DIR = _sample_cb.OUT_DIR\n"
        "    def _compat_normalizer_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):\n"
        "        if self.device is None:\n"
        "            self.device = batch.protein_pos.device\n"
        "            self.pos_normalizer = self.pos_normalizer.to(self.device)\n"
        "        batch.protein_pos = batch.protein_pos / self.pos_normalizer\n"
        "        batch.ligand_pos = batch.ligand_pos / self.pos_normalizer\n"
        "    NormalizerCallback.on_test_batch_start = _compat_normalizer_start\n"
        "    call(protein_path, ligand_path, "
        "ckpt_path='./checkpoints/molcraft_epoch26-val_loss5.96-mol_stable0.95-complete0.97.ckpt', "
        "num_samples=int(os.environ.get('MOLCRAFT_NUM_SAMPLES', '16')), "
        "sample_steps=int(os.environ.get('MOLCRAFT_SAMPLE_STEPS', '100')), "
        "sampling_strategy='end_back_pmf')\n",
    )
    source = source.replace(
        "    seed_everything(cfg.seed)\n",
        "    # Released weights accept 13 ligand inputs despite stale time metadata.\n"
        "    cfg.dynamics.time_emb_dim = 0\n"
        "    seed_everything(cfg.seed)\n",
    )
    source = source.replace(
        "    out_fn = 'output/0.sdf'\n"
        "    metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()\n"
        "    print(json.dumps(metrics, indent=4, cls=NpEncoder))\n",
        "    print(json.dumps({'generation_only': True, 'output_dir': OUT_DIR}, indent=4))\n",
    )
    sys.argv = [str(script)] + sys.argv[1:]
    scope = {"__name__": "__main__", "__file__": str(script)}
    exec(compile(source, str(script), "exec"), scope)


if __name__ == "__main__":
    main()
