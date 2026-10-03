# Targeted Linguistics Analysis of Sign Language Models with Minimal Translation Pairs
This repository contains research code, the dataset (ASL-MTP), and analysis scripts for the paper [Targeted Linguistics Analysis of Sign Language Models with Minimal Translation Pairs](http://arxiv.org/abs/2604.27232). 

<img src="images/fig1.png" width="600" alt="Figure 1" />

Models of sign language have historically lagged behind those for spoken language (text and speech). Recent work has greatly improved their performance on tasks like sign language translation and isolated sign recognition. However, it remains unclear to what extent existing models capture various linguistic phenomena of sign language, and how well they use cues from the multiple articulators used in sign language (hands, upper body, face). We introduce a new benchmark dataset for American Sign Language, ASL Minimal Translation Pairs (ASL-MTP), divided into multiple types of sign language phenomena and corresponding minimal pairs of translations, for performing such linguistic analyses. As a case study, we use ASL-MTP to analyze a state-of-the-art ASL-to-English translation model. We conduct a targeted analysis of the model by ablating various input cues during training and inference and evaluating on the phenomena in ASL-MTP. Our results show that, while the model performs above chance level on most of the phenomena, it relies strongly on manual cues while often missing crucial non-manual cues.

----
## Usage
### 1. ASL-MTP

The benchmark (ASL-MTP) with links to videos in [ASLLRP](https://dai.cs.rutgers.edu/dai/s/utterancesearchresult?16=2&inputSearch=&datasource=1539244&datasource=1379841&datasource=1379845&datasource=1512932&datasource=1512933&datasource=1512934&datasource=1379844&datasource=1379843&datasource=1512935&datasource=1379842&datasource=1379808&datasource=1379840&datasource=1379839&datasource=1379838&datasource=1379837&datasource=1379836&datasource=1512936&datasource=1512937&datasource=1379835&datasource=1379834&datasource=1379833&datasource=1379832&datasource=1379831&datasource=1379830&datasource=1379829&datasource=1512938&datasource=1379828&datasource=1379827&datasource=1379826&datasource=1379825&datasource=1379824&datasource=1512939&datasource=1379823&datasource=1379822&datasource=1379821&datasource=1379820&datasource=1379819&datasource=1379818&datasource=1379817&datasource=1379816&datasource=1379815&datasource=1379814&datasource=1379813&datasource=1379812&datasource=1379811&datasource=1379810&datasource=1379809&dspdatasource=1322367&dspdatasource=1322368&dspdatasource=1322384&dspdatasource=1322383&dspdatasource=1322382&dspdatasource=1235470&dspdatasource=1235767&dspdatasource=1235766&dspdatasource=1252521&dspdatasource=1235764&dspdatasource=1235763&dspdatasource=1252524&dspdatasource=1252520&dspdatasource=1235760&dspdatasource=1235759&dspdatasource=1249668&dspdatasource=1249401&dspdatasource=1252523&dspdatasource=1252522&dspdatasource=1235756&dspdatasource=1322381&dspdatasource=1322380&dspdatasource=1322379&dspdatasource=1322378&dspdatasource=1322377&dspdatasource=1322376&dspdatasource=1322375&dspdatasource=1322374&dspdatasource=1322373&dspdatasource=1322372&dspdatasource=1322371&dspdatasource=1322370&dspdatasource=1322369&rit3datasource=1542752&participant=23&participant=24&participant=6&participant=5&participant=2&participant=25&participant=26&participant=7&participant=8&participant=4&participant=27&participant=9&participant=10&participant=11&participant=1&participant=30&participant=3&participant=28&participant=12&participant=29&participant=13&video_views=noCare&color=noCare&sleeves=noCare&glasses=noCare&minOccur=-1&color=noCare&sleeves=noCare&glasses=noCare&allBox=noCare&results_view=front&resultsPerPage=25) is in [data](data/).

### 2. Model

The details about SHuBERT+ByT5, the training, and experiments are described in the paper.

### 3. Statistical Analysis

The analysis scripts and results are in [analysis](analysis/).

----
## Publications

Our preprint is accepted to the [CVPR 2026 GenSign, Generative AI for Sign Language Workshop](https://genai4sl.github.io/)!

Here are our [non-archival proceeding](https://genai4sl.github.io/assets/pdfs/05_Targeted_Linguistic_Analysis.pdf) and [poster](https://github.com/serpilkarabuklu/SL-Models-Analysis/blob/main/publications/Karabuklu_etal_CVPR%20Workshop%20ONLY%20Poster%20GenSign_final.pdf).

----
## Citing our work
If you find our work useful in your research, please consider citing:

```bibtex
@misc{karabüklü2026targetedlinguisticanalysissign,
      title={Targeted Linguistic Analysis of Sign Language Models with Minimal Translation Pairs}, 
      author={Serpil Karabüklü and Kanishka Misra and Shester Gueuwou and Diane Brentari and Greg Shakhnarovich and Karen Livescu},
      year={2026},
      eprint={2604.27232},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.27232}, 
}
```
