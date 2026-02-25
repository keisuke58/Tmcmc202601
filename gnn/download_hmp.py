#!/usr/bin/env python3
"""
HMP 口腔 16S データ取得 (Phase 2, Issue #39).

データソース:
- HMP16SData (Bioconductor/R): V13() 43,140 OTU × 2,898 samples, V35() 45,383 × 4,743
  - HMP_BODY_SUBSITE で oral をフィルタ
  - R + rpy2 または R で export → CSV を Python で読む
- HMPDACC: https://www.hmpdacc.org/HM16STR/ (直接ダウンロード)
- curatedMetagenomicData (Bioconductor): メタ解析用 20,000+ サンプル

Usage (R 経由の例):
  R -e "BiocManager::install('HMP16SData'); d=HMP16SData::V35(); ..."
  または rpy2 で Python から呼び出し

TODO:
- [ ] R/HMP16SData で oral サブセット取得
- [ ] 5 菌種 (So, An, Vd, Fn, Pg) への OTU マッピング
- [ ] co-occurrence network 構築
"""

import json
from pathlib import Path

# 5 菌種と HMP/NCBI  taxonomy の対応 (要調整)
SPECIES_MAPPING = {
    "S. oralis": ["Streptococcus", "oralis"],
    "A. naeslundii": ["Actinomyces", "naeslundii"],
    "V. dispar": ["Veillonella", "dispar"],
    "F. nucleatum": ["Fusobacterium", "nucleatum"],
    "P. gingivalis": ["Porphyromonas", "gingivalis"],
}


def get_hmp_sources():
    """HMP データソース一覧を返す。"""
    return {
        "HMP16SData": "Bioconductor R package. BiocManager::install('HMP16SData')",
        "HM16STR": "https://www.hmpdacc.org/HM16STR/ - 処理済み 16S",
        "HMR16S": "https://hmpdacc.org/hmp/hmp/HMR16S/ - Raw reads",
        "curatedMetagenomicData": "Bioconductor, 口腔サンプル含む 20k+",
    }


def main():
    print("HMP 口腔 16S データ取得 (Phase 2)")
    print("=" * 50)
    for name, desc in get_hmp_sources().items():
        print(f"  {name}: {desc}")
    print()
    print("次のステップ: R で HMP16SData::V35() を呼び、")
    print("  HMP_BODY_SUBSITE で oral をフィルタ → CSV export")
    print("  または rpy2 で Python から R を呼び出し")


if __name__ == "__main__":
    main()
