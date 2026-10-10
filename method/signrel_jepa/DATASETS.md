# Local data and split policy

No dataset video is copied into this project or downloaded from ASL-LEX.
Full manifests, landmark caches, weights, features and runs stay ignored by Git.

| Source | Current use | Local path | Terms/source |
|---|---|---|---|
| How2Sign train | label-free SSL; 100 random clips before structural preflight | `/home/shared_data/sign_language/How2Sign/train/raw_videos` | [How2Sign](https://how2sign.github.io/), CC BY-NC 4.0/research |
| ASL Citizen | separate frozen evaluation; official signer-independent splits | `/home/dongvk/datasets/ASL_Citizen` | [Dataset description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/); retain original access terms |
| WLASL | all source metadata excludes pretraining sources; RGB evaluation adapter | `/home/dongvk/datasets/WLASL2000/WLASL2000` | [WLASL](https://github.com/dxli94/WLASL), C-UDA/research |
| ASL-LEX | phonological table only, joined through official ASL Citizen codes | existing read-only CSV referenced in eval config | [ASL-LEX download](https://asl-lex.org/download.html), tabular metadata CC BY-NC 4.0; no reference video reuse |
| Sem-Lex / NMFs-CSL | available adapters accept user-provided licensed manifests; not present in local inventory | no assumed videos | Obtain the respective official access before experiments |

Sem-Lex access was rechecked on 2026-10-06 against the [official repository](https://github.com/leekezar/SemLex)
and its [download form](https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform).
The form requires a researcher's name, email, affiliation and agreement to data
terms before revealing downloads. No form was submitted and no identity or
institutional commitments were invented. Repository code licensing and the data
form's CC BY-NC-SA 4.0 terms are distinct. The permitted ASL Citizen + ASL-LEX
tabular join is the current phonological MVP evaluation; it is not a reproduced
Sem-Lex benchmark.
| YouTube-ASL | later scale candidate; local inventory contains IDs only | `/home/dongvk/datasets/youtubeASL` | no automated video downloader or captions in SSL |

Other observed local datasets (MS-ASL, CSLDaily, Phoenix14T) remain untouched.
The generic `common`/`video_folder` manifest import supports preparing RGB SSL
records without annotations. Dataset-specific segment timing must be checked
before a new corpus is used; no guessed keypoint time resampling is performed.

An SSL manifest contains only structural fields. `dataset`/`language` may be
stored for bookkeeping but are dropped by `ssl_record`; `SSLVideoDataset`,
collation and model forward all enforce semantic-label exclusion. ASL Citizen
labels and signer IDs enter only feature evaluation after SSL has finished.

YouTube source IDs are parsed from How2Sign filenames and WLASL URLs, including
short links. Exact source/path/URL/declared hashes are checked; optional
`--hash-files` computes video byte hashes. Any contaminated source is removed
entirely, including all its segments. Missing WLASL videos still participate in
the source exclusion manifest. A source-level clean report does not prove
absence of visually re-encoded duplicates; perceptual duplicate detection is an
optional extension, not an implemented audit claim.

The ASL Citizen pilot fixes 40 randomly chosen lexical classes before extracting
features, with 160 training/gallery, 80 validation/query and 80 test/query clips.
There is no signer overlap across official splits. All evaluation video sources,
including gallery sources, are excluded from local SignRel pretraining.
