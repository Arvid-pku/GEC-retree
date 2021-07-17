# Deep-Generative-Model-Project

Based on sequence modeling toolkit Fairseq (https://github.com/pytorch/fairseq)

## Requirements
- PyTorch version >= 1.4.0
- Python version >= 3.6

## Preprocess

```
python preprocess_graph.py --trainpref edge-data/train --validpref edge-data/valid \
--testpref edge-data/test--source-lang src --target-lang tgt --destdir data/bin \
--nwordssrc 50000 --workers 5 --edgedict edge-data/dict.edge.txt \
--task translation_with_graph_attention_with_copy
```
```
python process_graph_copy.py --testpref edge-data/test --source-lang src --target-lang tgt \
--destdir data/bin-copy  --nwordssrc 50000 --workers 5 \
--edgedict edge-data/dict.edge.txt --srcdict bin/dict.src.txt \
--tgtdict data/bin/dict.tgt.txt --dataset-impl raw
```

## Train

```
ARCH=transformer_concat_with_graph_copy_gigaword_big2

CUDA_VISIBLE_DEVICES=0 python train.py data/bin \
  -a $ARCH --optimizer adam --lr 0.0001 -s src -t tgt \
  --dropout 0.1 --max-tokens 2048 \
  --share-decoder-input-output-embed \
  --task translation_with_graph_attention_with_copy \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer-graph-copy \
  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --criterion cross_entropy_copy --update-freq 2

```

## Test
```
CUDA_VISIBLE_DEVICES=0  python generate.py data/bin-copy \
--task $ARCH  \
--path  checkpoints/$ARCH/checkpoint_best.pt \
--batch-size 128 --beam 5 --lenpen 1.2 --replace-unk --raw-text \
> output/transformer-graph-copy-concat2/conll14st-test.tok.trg 

```

