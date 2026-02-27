# RQ1: Cultural Uncertainty Avoidance and Fear Sentiment in Emerging Technology Stocks

## Research Question
Does Hofstede's uncertainty avoidance (UAI) amplify fear-related media sentiment toward emerging technology companies, particularly during AI disruption events?

## Data Sources
- **MarketPsych/LSEG Refinitiv RMA**: Daily sentiment for 178 tech companies across 8 countries (1998-2026)
- **Hofstede Cultural Dimensions**: 6D model (PDI, IDV, MAS, UAI, LTO, IVR)
- **AI Disruption Events**: 12 major events (ChatGPT launch, DeepSeek, NVIDIA surge, etc.)

## Files
- `analysis.py` - Full analysis pipeline (data loading, panel regression, event study, visualizations)
- `write_paper.py` - Generates the AMCIS 2026 ERF paper in Word format
- `AMCIS_2026_ERF_Paper_RQ1.docx` - The generated paper

## Key Findings
1. UAI significantly predicts fear sentiment (beta=0.050, p<0.05)
2. UAI x Buzz interaction is significant (beta=-0.036, p<0.01)
3. High-UAI countries show 6x larger abnormal fear during AI events
4. Results robust across multiple Hofstede dimension controls

## How to Run
```bash
pip install pandas numpy statsmodels scipy linearmodels matplotlib seaborn requests python-docx
python analysis.py
python write_paper.py
```

## Sample: 923,524 observations, 178 firms, 8 countries
