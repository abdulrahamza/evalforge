# EvalForge
A backend testing and evaluation system for AI agents.

## Overview
EvalForge is a rigorous testing environment designed to evaluate the performance of AI agents. As AI systems become more complex and autonomous, establishing standardized, objective benchmarks for agent reliability is critical. EvalForge solves this by providing a unified API to subject models to various test scenarios and programmatically score their behavior, tool execution, and strategic reasoning.

## Current Features
- **Scenario-Based Evaluation:** Submit agents to parameterized testing scenarios to measure adaptability.
- **Scoring System:** Quantitative performance metrics across reasoning, task execution, and tool use.
- **RESTful API:** FastAPI-based backend for submitting evaluation jobs and retrieving results.
- **Structured Outputs:** Evaluation results returned in standardized JSON for predictable downstream consumption.

## Example Output
```json
{
  "agent_id": "agn_09x4a2",
  "evaluation_run": "run_7781x",
  "scenarios_tested": 10,
  "scores": {
    "reasoning": 88.5,
    "tool_use": 92.0,
    "task_execution": 85.0,
    "overall": 88.5
  },
  "metadata": {
    "planned_nft_schema": {
      "name": "EvalForge Performance Certificate",
      "attributes": [
        { "trait_type": "Overall Score", "value": 88.5 },
        { "trait_type": "Tier", "value": "A" }
      ],
      "ipfs_uri": "pending_phase_2"
    }
  }
}
```

## How It Works
1. **Initialize:** The developer registers their agent's endpoint and description with the EvalForge API.
2. **Generate Scenarios:** EvalForge creates targeted, edge-case, and stress-test scenarios tailored to the agent's expected capabilities.
3. **Execute:** The backend triggers the agent against these scenarios, tracking latency, responses, and API interactions.
4. **Evaluate & Score:** Results are aggregated and assigned categorical scores via an LLM judge, producing the final evaluation JSON.

## Getting Started
### Prerequisites
- Python 3.10+
- Anthropic API Key (or supported LLM provider)

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/abdulrahamza/evalforge.git
   cd evalforge
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   Create a `.env` file and add your keys:
   ```env
   ANTHROPIC_API_KEY=your_key_here
   ```
4. Run the API:
   ```bash
   uvicorn main:app --reload
   ```

## Roadmap

### Phase 1 (Completed)
- Core FastAPI backend implementation
- Adversarial scenario generation
- LLM-judged scoring mechanism
- JSON result aggregation

### Phase 2 (Beam Integration - Planned)
- **NFT Minting (ERC-721):** Mint agent evaluation scorecards as on-chain assets.
- **IPFS Storage:** Store permanent JSON evaluation records on decentralized storage.
- **Scorecard System:** Immutable, verifiable proof of an AI agent's performance capabilities at a given point in time.

### Phase 3 (Advanced Features - Planned)
- **Staking:** Stake tokens on agent performance predictions.
- **Oracle-Based Updates:** Dynamic performance state updates feeding directly into on-chain registries via oracles.

## Beam Integration (Planned)
EvalForge intends to utilize the Beam network to establish an immutable, verifiable registry of AI agent performance. Beam provides the high-throughput, low-cost environment necessary for minting continuous evaluation certificates (NFTs) without friction. By persisting evaluation data on Beam, developers can mathematically prove their agent's capabilities to smart contracts, marketplaces, or users, bridging off-chain execution with on-chain trust.

## Vision
The goal is to position EvalForge as the standardized performance layer for the AI agent economy. As agents increasingly interact, trade, and execute tasks autonomously, a verifiable, trustless source of truth regarding their operational reliability will be required.

## Author
**Founder:** hamxa (X: [@abdulrahamza](https://x.com/abdulrahamza))
- Solo builder
- Previous Work: FlowPay
