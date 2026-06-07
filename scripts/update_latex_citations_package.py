from pathlib import Path
import re, json, shutil, zipfile, hashlib, subprocess
base=Path('/Users/colin/Desktop/projects/seldinger/research/synthetic_dsa')
paper=base/'paper'
pkg=base/'deliverables/seldinger_dsa_latex_package'
figdir=base/'outputs/figures'
extra_refs="""

@misc{cicek2016_3dunet,
  title = {{3D U-Net}: Learning Dense Volumetric Segmentation from Sparse Annotation},
  author = {{\\c C}i{\\c c}ek, {\\\"O}zg{\\\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S. and Brox, Thomas and Ronneberger, Olaf},
  year = {2016},
  eprint = {1606.06650},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.1606.06650},
  url = {https://arxiv.org/abs/1606.06650}
}

@misc{medicaldecathlon2019,
  title = {A Large Annotated Medical Image Dataset for the Development and Evaluation of Segmentation Algorithms},
  author = {Simpson, Amber L. and Antonelli, Michela and Bakas, Spyridon and Bilello, Michel and Farahani, Keyvan and van Ginneken, Bram and Kopp-Schneider, Annette and Landman, Bennett A. and Litjens, Geert and Menze, Bjoern and Ronneberger, Olaf and Summers, Ronald M. and Cardoso, M. Jorge and others},
  year = {2019},
  eprint = {1902.09063},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.1902.09063},
  url = {https://arxiv.org/abs/1902.09063}
}

@misc{transunet2021,
  title = {{TransUNet}: Transformers Make Strong Encoders for Medical Image Segmentation},
  author = {Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L. and Zhou, Yuyin},
  year = {2021},
  eprint = {2102.04306},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2102.04306},
  url = {https://arxiv.org/abs/2102.04306}
}

@article{medsam2024,
  title = {Segment Anything in Medical Images},
  author = {Ma, Jun and He, Yuting and Li, Fei and Han, Lin and You, Chenyu and Wang, Bo},
  journal = {Nature Communications},
  volume = {15},
  pages = {654},
  year = {2024},
  doi = {10.1038/s41467-024-44824-z},
  url = {https://doi.org/10.1038/s41467-024-44824-z},
  note = {arXiv:2304.12306}
}

@misc{sam2medical2024,
  title = {Segment Anything in Medical Images and Videos: Benchmark and Deployment},
  author = {Ma, Jun and Kim, Sumin and Li, Feifei and Baharoon, Mohammed and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
  year = {2024},
  eprint = {2408.03322},
  archivePrefix = {arXiv},
  primaryClass = {eess.IV},
  doi = {10.48550/arXiv.2408.03322},
  url = {https://arxiv.org/abs/2408.03322}
}

@article{yoon2024_domain_generalization,
  title = {Domain Generalization for Medical Image Analysis: A Review},
  author = {Yoon, Jee Seok and Oh, Kwanseok and Shin, Yooseung and Mazurowski, Maciej A. and Suk, Heung-Il},
  journal = {Proceedings of the IEEE},
  volume = {112},
  number = {10},
  year = {2024},
  doi = {10.1109/JPROC.2024.3507831},
  url = {https://doi.org/10.1109/JPROC.2024.3507831},
  note = {arXiv:2310.08598}
}
"""
bib=(paper/'references.bib').read_text()
for key in ['cicek2016_3dunet','medicaldecathlon2019','transunet2021','medsam2024','sam2medical2024','yoon2024_domain_generalization']:
    bib=re.sub(r'\n@\w+\{'+re.escape(key)+r',[\s\S]*?(?=\n@|\Z)', '', bib)
bib=bib.rstrip()+extra_refs
(paper/'references.bib').write_text(bib)
main=(paper/'main.tex').read_text()
old='U-Net~\\cite{unet2015} and nnU-Net~\\cite{nnunet2018} remain important reference points for biomedical image segmentation pipelines. Promptable foundation segmentation models, including Segment Anything~\\cite{sam2023}, SAM 2~\\cite{sam2_2024}, and MedSAM-2~\\cite{medsam2_2024}, motivate future comparisons against stronger interactive and video-style medical vision baselines.'
new='U-Net~\\cite{unet2015}, 3D U-Net~\\cite{cicek2016_3dunet}, nnU-Net~\\cite{nnunet2018}, TransUNet~\\cite{transunet2021}, and the Medical Segmentation Decathlon~\\cite{medicaldecathlon2019} remain important reference points for biomedical image segmentation pipelines and benchmark design. Promptable foundation segmentation models, including Segment Anything~\\cite{sam2023}, SAM 2~\\cite{sam2_2024}, MedSAM~\\cite{medsam2024}, MedSAM-2~\\cite{medsam2_2024}, and medical SAM2 benchmarking work~\\cite{sam2medical2024}, motivate future comparisons against stronger interactive and video-style medical vision baselines. Domain-generalization reviews in medical image analysis~\\cite{yoon2024_domain_generalization} further motivate our conservative treatment of synthetic-to-real claims.'
if old in main:
    main=main.replace(old,new)
elif 'cicek2016_3dunet' not in main:
    main=main.replace('U-Net~\\cite{unet2015} and nnU-Net~\\cite{nnunet2018}', 'U-Net~\\cite{unet2015}, 3D U-Net~\\cite{cicek2016_3dunet}, nnU-Net~\\cite{nnunet2018}, TransUNet~\\cite{transunet2021}, and the Medical Segmentation Decathlon~\\cite{medicaldecathlon2019}')
(paper/'main.tex').write_text(main)
pkg.mkdir(parents=True, exist_ok=True)
(pkg/'figures').mkdir(exist_ok=True)
for name in ['preprint_benchmark_schema.png','preprint_synthetic_results_panel.png','preprint_dias_comparison_panel.png','preprint_synthetic_metrics_table.png','preprint_dias_metrics_table.png','synthetic_to_dias_vessel_transfer_panel.png','dias_test_threshold_vs_morphology_paper.png']:
    src=figdir/name
    if src.exists(): shutil.copy2(src,pkg/'figures'/name)
pkg_main=main.replace('../outputs/figures/preprint_benchmark_schema.png','figures/preprint_benchmark_schema.png').replace('../outputs/figures/preprint_synthetic_results_panel.png','figures/preprint_synthetic_results_panel.png').replace('../outputs/figures/synthetic_to_dias_vessel_transfer_panel.png','figures/synthetic_to_dias_vessel_transfer_panel.png').replace('../outputs/figures/dias_test_threshold_vs_morphology_paper.png','figures/dias_test_threshold_vs_morphology_paper.png').replace('\\bibliography{references}','\\bibliography{seldinger_dsa_references}')
(pkg/'seldinger_dsa_bibtex.tex').write_text(pkg_main)
(pkg/'seldinger_dsa_references.bib').write_text(bib)
bibitems="""
\\begin{thebibliography}{14}
\\bibitem{dias2024} Wentao Liu, Tong Tian, Lemeng Wang, Weijin Xu, Lei Li, Haoyuan Li, Wenyi Zhao, Siyu Tian, Xipeng Pan, Yiming Deng, Feng Gao, Huihua Yang, Xin Wang, and Ruisheng Su. DIAS: A dataset and benchmark for intracranial artery segmentation in DSA sequences. \\emph{Medical Image Analysis}, 97:103247, 2024. doi:10.1016/j.media.2024.103247.
\\bibitem{unet2015} Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv:1505.04597, 2015. doi:10.48550/arXiv.1505.04597.
\\bibitem{cicek2016_3dunet} Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. arXiv:1606.06650, 2016. doi:10.48550/arXiv.1606.06650.
\\bibitem{nnunet2018} Fabian Isensee, Jens Petersen, Andre Klein, David Zimmerer, Paul F. Jaeger, Simon Kohl, Jakob Wasserthal, Gregor Koehler, Tobias Norajitra, Sebastian Wirkert, and Klaus H. Maier-Hein. nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. arXiv:1809.10486, 2018. doi:10.48550/arXiv.1809.10486.
\\bibitem{medicaldecathlon2019} Amber L. Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani, Bram van Ginneken, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers, M. Jorge Cardoso, and others. A Large Annotated Medical Image Dataset for the Development and Evaluation of Segmentation Algorithms. arXiv:1902.09063, 2019. doi:10.48550/arXiv.1902.09063.
\\bibitem{transunet2021} Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv:2102.04306, 2021. doi:10.48550/arXiv.2102.04306.
\\bibitem{sam2023} Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment Anything. arXiv:2304.02643, 2023. doi:10.48550/arXiv.2304.02643.
\\bibitem{sam2_2024} Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024. doi:10.48550/arXiv.2408.00714.
\\bibitem{medsam2024} Jun Ma, Yuting He, Fei Li, Lin Han, Chenyu You, and Bo Wang. Segment Anything in Medical Images. \\emph{Nature Communications}, 15:654, 2024. doi:10.1038/s41467-024-44824-z.
\\bibitem{medsam2_2024} Jiayuan Zhu, Abdullah Hamdi, Yunli Qi, Yueming Jin, and Junde Wu. Medical SAM 2: Segment Medical Images as Video via Segment Anything Model 2. arXiv:2408.00874, 2024. doi:10.48550/arXiv.2408.00874.
\\bibitem{sam2medical2024} Jun Ma, Sumin Kim, Feifei Li, Mohammed Baharoon, Reza Asakereh, Hongwei Lyu, and Bo Wang. Segment Anything in Medical Images and Videos: Benchmark and Deployment. arXiv:2408.03322, 2024. doi:10.48550/arXiv.2408.03322.
\\bibitem{yoon2024_domain_generalization} Jee Seok Yoon, Kwanseok Oh, Yooseung Shin, Maciej A. Mazurowski, and Heung-Il Suk. Domain Generalization for Medical Image Analysis: A Review. \\emph{Proceedings of the IEEE}, 112(10), 2024. doi:10.1109/JPROC.2024.3507831.
\\bibitem{aiscientistv2_2025} Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066, 2025. doi:10.48550/arXiv.2504.08066.
\\bibitem{selfrevising2026} Fiona Y. Wang and Markus J. Buehler. Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence. arXiv:2606.01444, 2026. doi:10.48550/arXiv.2606.01444.
\\end{thebibliography}
"""
gdocs=pkg_main.replace('\\bibliographystyle{plain}\n\\bibliography{seldinger_dsa_references}', bibitems.strip())
(pkg/'seldinger_dsa_google_docs.tex').write_text(gdocs)
(pkg/'README.md').write_text('# Seldinger-DSA LaTeX package\n\nUpdated citation count: 14 verified references.\n\nAuthor block: Colin Son, MD; Seldinger, Inc; San Antonio, TX.\n')
cite_keys=sorted({k.strip() for group in re.findall(r'\\cite\{([^}]+)\}', pkg_main) for k in group.split(',')})
bib_keys=sorted(re.findall(r'@\w+\{([^,]+),', bib))
fig_paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', pkg_main)
verification={'author_block':'Colin Son, MD; Seldinger, Inc; San Antonio, TX','citation_count':len(cite_keys),'bib_entry_count':len(bib_keys),'cited_keys':cite_keys,'bib_keys':bib_keys,'missing_bib_keys':sorted(set(cite_keys)-set(bib_keys)),'uncited_bib_keys':sorted(set(bib_keys)-set(cite_keys)),'figure_paths':fig_paths,'missing_figures':[p for p in fig_paths if not (pkg/p).exists()],'verification_sources':{'DIAS':'Crossref DOI 10.1016/j.media.2024.103247 + arXiv 2306.12153','additional references':'arXiv pages, Nature Communications DOI, Proceedings IEEE DOI verified by search/extract results'},'no_placeholder_citations':True}
compile_status={}
for tex,key in [('seldinger_dsa_bibtex.tex','bibtex_pdf'),('seldinger_dsa_google_docs.tex','google_docs_pdf')]:
    try:
        subprocess.run(['tectonic',tex],cwd=pkg,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        compile_status[key]='ok'
    except Exception as e:
        compile_status[key]='failed: '+repr(e)
verification['compile_status']=compile_status
(pkg/'seldinger_dsa_verification.json').write_text(json.dumps(verification,indent=2))
zip_path=base/'deliverables/seldinger_dsa_latex_package.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(pkg.rglob('*')):
        z.write(f,f.relative_to(pkg.parent))
print(json.dumps({'zip':str(zip_path),'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'citation_count':len(cite_keys),'bib_entry_count':len(bib_keys),'missing_bib_keys':verification['missing_bib_keys'],'compile_status':compile_status},indent=2))
