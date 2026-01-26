# CDT v2 Prototypes

**Status: Work in Progress**

This folder contains prototype notebooks for CDT v2 features. These are experimental implementations and may change significantly before the final v2 release.

## CDT_v2_RawExpression_Prototype.ipynb

**Purpose**: Process raw scRNA-seq data directly without RNA Language Models (e.g., scGPT)

**Key Features**:
- Raw Expression Encoder: directly processes scRNA-seq counts (CPM + log1p normalized)
- No dependency on external foundation models
- Flash Attention for efficient GPU memory usage
- Validated performance: Pearson r = 0.5034

**Why This Approach?**
- Many experimentalists don't have a suitable pre-trained model for their specific cell type or condition
- Direct interpretability: understand your data without black-box transformations
- Lower barrier to entry: no need to run large foundation models

---

## CDT_v2_DNARNA_Prototype.ipynb

**Purpose**: Protein-free CDT using only DNA + RNA (no Protein Language Model required)

**Key Features**:
- DNA (Enformer) + Raw RNA expression only
- Removed: Protein projector, Protein self-attention, RNA→Protein cross-attention
- VCE: 2-modality pooling (DNA + RNA)
- Validated performance: **Pearson r = 0.5022** (same as 3-modality version!)
- ~22% parameter reduction (44M vs 57M)

**Why This Approach?**
- Most experimentalists don't have Proteomics data
- Proteomics experiments are expensive and time-consuming
- **Proof**: Protein LM is NOT required for CRISPRi effect prediction

**Architecture**:
```
DNA [896, 3072] → Projector → Self-Attn(2 layers)
                      ↓ Cross-Attn
RNA [2360] → RawExpressionEncoder → Self-Attn(1 layer)
                      ↓
                    VCE (2-modality pooling)
                      ↓
                    Task Layer → [n_genes]
```

---

**Note**: These are prototypes. The API and implementation may change.
