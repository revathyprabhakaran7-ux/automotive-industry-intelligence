
# Automotive Industry Intelligence

### Business Intelligence | Machine Learning | AI Engineering

A portfolio of automotive industry projects exploring how business intelligence, predictive analytics, machine learning, and artificial intelligence can support data-driven business decisions.

The portfolio combines commercial business analysis with modern data engineering, MLOps, and AI engineering practices.

Projects use publicly available automotive industry data, reproducible analytical workflows, model evaluation, and interactive applications.

## Projects

### 01 — Automotive Business Intelligence

**Status: Version 1 Completed**

**[View Interactive Tableau Dashboard](https://public.tableau.com/app/profile/revathy.prabhakaran/viz/CanadianAutomotiveSalesIntelligence/CanadianAutomotiveMarketPerformance)**

A business intelligence project analyzing Canadian automotive sales performance using official Statistics Canada data.

#### Business Objectives

- Monitor monthly new vehicle sales
- Analyze month-over-month and year-over-year growth
- Identify demand trends and seasonal fluctuations
- Examine historical automotive market performance
- Communicate business insights through interactive dashboards

#### Implementation

- Downloaded official Statistics Canada automotive sales data
- Cleaned and validated data using Python and pandas
- Developed an automated data-processing pipeline with GitHub Actions
- Calculated monthly growth and rolling 12-month KPIs
- Built an interactive Tableau Public dashboard
- Documented business findings and analytical limitations

#### Key Performance Indicators — August 2026

| KPI | Value |
|---|---:|
| Latest Monthly Vehicle Sales | 174,611 |
| Year-over-Year Growth | +4.86% |
| Month-over-Month Growth | -0.84% |
| Rolling 12-Month Sales | 1,940,113 |
| Rolling 12-Month Average | 161,676 |

#### Business Insights

Canadian new vehicle sales increased approximately 4.86% compared with August 2025, while declining 0.84% from July 2026.

Historical analysis highlights recurring seasonal demand patterns and significant market disruptions, including the decline in 2020 and subsequent rebound.

The rolling 12-month metrics provide a broader view of automotive market demand beyond individual monthly fluctuations.

#### Technologies

Python, pandas, Tableau Public, GitHub Actions, Git

**Data Source:** Statistics Canada, Table 20-10-0085-01

**Current Dataset:** January 1946 – August 2026

**[Explore Project 01](01-business-intelligence/)**

---

### 02 — Automotive Sales Forecasting and MLOps

**Status: Planned**

A machine learning project designed to forecast automotive sales demand using historical Canadian vehicle sales data.

#### Business Objectives

- Analyze historical demand patterns
- Identify trends and seasonality
- Forecast future automotive sales
- Compare forecasting models
- Evaluate predictive performance
- Support data-driven sales planning

#### Planned Implementation

1. Exploratory data analysis
2. Time-series preprocessing and feature engineering
3. Baseline forecasting
4. Machine learning model development
5. Model comparison and evaluation
6. Experiment tracking with MLflow
7. Model versioning
8. Automated training and evaluation workflows
9. Forecast visualization and business interpretation

#### Model Evaluation

Models will be evaluated using chronological data splits and appropriate forecasting metrics.

Planned metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE), where appropriate

Model results will be compared against baseline forecasts to assess whether additional complexity improves predictive performance.

#### Technologies

Python, pandas, NumPy, scikit-learn, MLflow, GitHub Actions, Git, MLOps

---

### 03 — Automotive AI Analytics Assistant

**Status: Planned**

An AI-powered analytics application that enables business users to explore automotive market data through natural-language conversations.

The project will integrate generative AI with structured data retrieval and analytical tools.

#### Business Objectives

- Enable natural-language exploration of automotive sales data
- Retrieve accurate sales statistics
- Explain market trends and growth patterns
- Generate data-grounded analytical summaries
- Support business users without requiring SQL or Python knowledge

#### Planned Implementation

1. Develop Python-based automotive analytics tools
2. Create a Model Context Protocol (MCP) server
3. Expose structured sales data and analytical functions through MCP tools
4. Integrate the tools with an AI agent
5. Use Azure AI Foundry for AI application development
6. Implement LLM-based natural-language interaction
7. Enable agent tool calling
8. Retrieve and calculate results using real automotive data
9. Evaluate response accuracy and tool-use behavior
10. Implement tracing and observability
11. Document the application architecture and evaluation results

#### Planned Capabilities

- Natural-language business questions
- MCP-based data access
- AI agent tool calling
- Automated analytical calculations
- Structured data retrieval
- Context-aware business explanations
- Grounded responses based on source data
- AI evaluation and tracing

The assistant will use programmatic calculations for numerical results rather than rely on the language model to generate unsupported statistics.

#### Technologies

Python, Azure AI Foundry, Large Language Models, MCP, AI Agents, Tool Calling, Prompt Engineering, Evaluation, Tracing, GitHub Actions

---

## Technology Stack

### Business Intelligence and Data Analytics

- Python
- pandas
- NumPy
- SQL
- Tableau
- Excel
- KPI Development
- Data Visualization

### Machine Learning and MLOps

- scikit-learn
- Time-Series Forecasting
- Feature Engineering
- Model Evaluation
- MLflow
- Model Versioning
- GitHub Actions
- Automated Training and Evaluation

### Generative AI and AI Engineering

- Azure AI Foundry
- Large Language Models
- Prompt Engineering
- AI Agents
- Function and Tool Calling
- Model Context Protocol (MCP)
- AI Evaluation
- Tracing and Observability

### Development and Automation

- Git
- GitHub
- GitHub Actions
- Python Automation
- CI/CD
- Technical Documentation

### Project Management

- Agile/Scrum
- Jira
- Requirements Analysis
- Technical Documentation

Technologies listed include tools already used in completed work and tools planned for upcoming projects. Each project identifies its implementation status.

---

## Data Sources

The portfolio uses publicly available automotive industry datasets.

### Primary Source

**Statistics Canada — Table 20-10-0085-01**

The current dataset contains:

- Canadian new motor vehicle sales
- All vehicle categories combined
- All fuel types
- Total country of manufacture
- Sales measured in units
- Unadjusted monthly observations

Additional data sources will be documented as projects develop.

---

## Data Engineering and Automation

Project 01 includes a Python-based data pipeline automated using GitHub Actions.

### Pipeline

1. Download Statistics Canada data
2. Filter relevant vehicle sales records
3. Clean and validate observations
4. Standardize dates and numerical values
5. Calculate business performance indicators
6. Export processed datasets
7. Upload CSV outputs as GitHub Actions artifacts

### Python Scripts

Located in `01-business-intelligence/src/`:

- `load_data.py` — downloads source data
- `prepare_data.py` — prepares monthly vehicle sales data
- `calculate_kpis.py` — calculates business intelligence KPIs

### Generated Outputs

- `monthly_vehicle_sales.csv`
- `vehicle_sales_kpis.csv`

### GitHub Actions

Workflow:

`.github/workflows/download-data.yml`

The workflow can be manually triggered to generate updated CSV artifacts.

The published Tableau dashboard uses a separately uploaded data extract and requires a data refresh to reflect newer pipeline outputs.

---

## Repository Structure

```text
automotive-industry-intelligence/
│
├── .github/
│   └── workflows/
│       └── download-data.yml
│
├── 01-business-intelligence/
│   ├── src/
│   │   ├── load_data.py
│   │   ├── prepare_data.py
│   │   └── calculate_kpis.py
│   ├── data/
│   └── outputs/
│
└── README.md
```

Folders and source code for Projects 02 and 03 will be added during implementation.

---

## Development Approach

Each project follows a structured development process:

1. Define the business problem
2. Identify reliable data sources
3. Explore and validate the data
4. Develop the analytical or AI solution
5. Evaluate results
6. Automate workflows where appropriate
7. Visualize and communicate findings
8. Document limitations and future improvements

The emphasis is on combining technical implementation with practical commercial value.

---

## Portfolio Objectives

This portfolio demonstrates how data analytics, predictive modeling, and AI engineering can address automotive business challenges.

Areas of focus include:

- Automotive market intelligence
- Sales performance analysis
- Demand forecasting
- Data pipeline automation
- Machine learning and MLOps
- AI agents and MCP integration
- AI-powered analytical decision support

---

## Project Status

| Project | Status |
|---|---|
| 01 — Automotive Business Intelligence | Version 1 Completed |
| 02 — Automotive Sales Forecasting and MLOps | Planned |
| 03 — Automotive AI Analytics Assistant | Planned |

The portfolio will be expanded with forecasting models, MLflow experiments, MCP tools, AI agent implementations, evaluation results, and supporting documentation.

---

**Portfolio Focus:** Automotive Analytics | Business Intelligence | Machine Learning | MLOps | AI Engineering
