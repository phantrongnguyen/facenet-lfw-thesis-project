# Face Model Benchmark 0.0.6A

Generated: 2026-05-22 14:50 +07:00

## Summary table

| pipeline                         |   embeddings |   people |   embedding_dim |   pairs |   load_seconds |   compare_seconds |   compare_ms_per_pair |   threshold |   accuracy |    far |    frr |   tp |   tn |   fp |   fn |
|:---------------------------------|-------------:|---------:|----------------:|--------:|---------------:|------------------:|----------------------:|------------:|-----------:|-------:|-------:|-----:|-----:|-----:|-----:|
| Facenet512_retinaface_embeddings |         7701 |     4281 |             512 |   10000 |      0.145278  |         0.0242223 |            0.00242223 |    0.407157 |     0.9613 | 0.0216 | 0.0558 | 4721 | 4892 |  108 |  279 |
| Facenet_retinaface_embeddings    |         7701 |     4281 |             128 |   10000 |      0.0443943 |         0.0063737 |            0.00063737 |    0.401176 |     0.9545 | 0.0214 | 0.0696 | 4652 | 4893 |  107 |  348 |
| Facenet512_mtcnn_embeddings      |         7701 |     4281 |             512 |   10000 |      0.144058  |         0.0265612 |            0.00265612 |    0.417026 |     0.9476 | 0.0196 | 0.0852 | 4574 | 4902 |   98 |  426 |
| Facenet512_embeddings            |         7696 |     4280 |             512 |   10000 |      0.039407  |         0.0243767 |            0.00243767 |    0.420161 |     0.9469 | 0.018  | 0.0882 | 4559 | 4910 |   90 |  441 |
| Facenet_embeddings               |         7696 |     4280 |             128 |   10000 |      0.0192605 |         0.0066254 |            0.00066254 |    0.391451 |     0.9385 | 0.027  | 0.096  | 4520 | 4865 |  135 |  480 |
| Facenet_mtcnn_embeddings         |         7701 |     4281 |             128 |   10000 |      0.0598243 |         0.0057735 |            0.00057735 |    0.366843 |     0.9333 | 0.0404 | 0.093  | 4535 | 4798 |  202 |  465 |

## Selected pipeline

**Recommended:** `Facenet512_retinaface_embeddings`

- Accuracy: `0.9613`
- FAR: `0.0216`
- FRR: `0.0558`
- Threshold: `0.407157`
- Compare speed: `0.002422 ms/pair`

## Decision note

For attendance, low FAR is important because false acceptance can create wrong attendance records. The selected pipeline has the best verification accuracy in this embedding benchmark. Before 0.0.6B integration, use this threshold as the initial value and validate again with real webcam/classroom images.