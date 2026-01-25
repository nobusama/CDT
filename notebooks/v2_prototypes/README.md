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

**Note**: This is a prototype. The API and implementation may change.
