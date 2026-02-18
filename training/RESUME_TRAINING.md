# RESUME TRAINING GUIDE

## What Was Completed

1. **Downloaded Expresso Dataset**: 11,615 samples in `training/data/expresso/`
2. **Extracted Talia Speaker**: 2,312 high-quality samples in `training/data/expresso_talia/`
3. **Preprocessed for CSM**: 2,081 train / 231 val samples in `training/data/csm_ready_ex04/`
4. **Pre-tokenized Audio**: All audio tokenized with Mimi in `training/data/csm_ready_ex04/tokens/`

## Ready to Train

All data is prepared. Just run:

```bash
cd /home/ec2-user/SageMaker/project_maya
python training/scripts/04_train_csm_efficient.py \
    --data training/data/csm_ready_ex04 \
    --epochs 25 \
    --lr 3e-5 \
    --output training/checkpoints/csm_maya
```

## Training Configuration

- **Model**: CSM-1B (sesame/csm-1b)
- **Dataset**: 2,081 train samples, 231 val samples (Talia voice, ~2 hours)
- **Epochs**: 25
- **Learning Rate**: 3e-5
- **Effective Batch Size**: 16 (batch=1, accumulation=16)
- **Precision**: BF16

## After Training

1. Evaluate:
```bash
python training/scripts/05_evaluate_model.py --checkpoint training/checkpoints/csm_maya/best_model
```

2. Integrate:
```bash
python training/scripts/06_integrate_model.py --checkpoint training/checkpoints/csm_maya/best_model
```

3. Test Maya:
```bash
python run.py
```

## Files Created

- `training/scripts/01_download_expresso.py` - Download dataset
- `training/scripts/02_extract_single_speaker.py` - Extract speaker
- `training/scripts/03_preprocess_audio.py` - Preprocess audio
- `training/scripts/03b_tokenize_audio.py` - Pre-tokenize with Mimi
- `training/scripts/04_train_csm_efficient.py` - Memory-efficient training
- `training/scripts/05_evaluate_model.py` - Evaluation
- `training/scripts/06_integrate_model.py` - Integration
- `training/configs/default.yaml` - Training config
- `training/run_pipeline.sh` - Full pipeline script

## Why Previous Training Failed

A root-owned background process (PID 27648) was using 10.2GB GPU memory.
After instance restart, this process should be gone.

## Quality Targets

- UTMOS: > 4.0 (naturalness)
- Speaker Similarity: > 0.85
- CER: < 5%
- WER: < 10%
