"""HPC agent — cluster resource query, Slurm refinement, job submission.

Pre-computation (always pre — HPC doesn't parse computation output):
- query_resources: mock squeue/sinfo for cluster state
- refine_slurm: merge rough Slurm + cluster info → final script
- submit: mock sbatch, return job_id

Cleanup before retry: scancel all jobs in batch.
"""
from langgraph.graph import StateGraph, START, END

from contracts.hpc_task import HPCParams, HPCResult
from contracts.agent_task import Artifact, TaskStatus
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class HPCState(BaseAgentState[HPCParams, HPCResult]):
    pass


# ═══════════════════════════════════════════════════════════════════

def query_resources(state: HPCState) -> dict:
    """Query cluster state via hpc-mcp queue-status tool.

    Returns node list with CPU/memory availability, queue depth.
    Mock: returns a single local node with 64 cores.

    TODO: HPC engineer wires real squeue/sinfo parsing.
    """
    cluster_info = {
        "nodes": [{"name": "node01", "cpus_avail": 64, "mem_avail_mb": 256000}],
        "queue_depth": 0,
        "partition": "compute",
    }
    return {
        "node_history": state.node_history + ["query_resources"],
        "scratchpad": {**state.scratchpad, "cluster_info": cluster_info},
    }


def refine_slurm(state: HPCState) -> dict:
    """Merge rough Slurm templates with cluster resource info.

    Adds: partition, account, node constraints, email notifications.
    Mock: appends partition=compute and account=default.

    TODO: HPC engineer wires actual template merging logic.
    """
    cluster = state.scratchpad.get("cluster_info", {})
    partition = cluster.get("partition", "compute")

    # In production: read rough slurm via ArtifactResolver, refine, write final
    refined_scripts = {}
    for job in state.task.params.jobs if state.task else []:
        refined = f"""#!/bin/bash
#SBATCH --job-name={job.agent}-{state.task_id}
#SBATCH --partition={partition}
#SBATCH --account=default
#SBATCH --mem={job.mem_per_cpu_mb}
#SBATCH --cpus-per-task={job.nprocs}
#SBATCH --time={job.walltime_hours}:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Refined by Tower HPC agent

# TODO: actual compute command from rough template
"""
        refined_scripts[job.agent] = refined

    return {
        "node_history": state.node_history + ["refine_slurm"],
        "scratchpad": {**state.scratchpad, "refined_scripts": refined_scripts},
    }


def submit(state: HPCState) -> dict:
    """Submit jobs via sbatch. Mock: returns fake job IDs.

    WARNING: NOT idempotent. Each call creates new jobs.
    Retry requires scancel first.

    TODO: HPC engineer wires real sbatch + job ID extraction.
    """
    import uuid

    job_ids = {}
    for agent_name in state.scratchpad.get("refined_scripts", {}):
        job_ids[agent_name] = str(uuid.uuid4().int % 90000 + 10000)

    return {
        "node_history": state.node_history + ["submit"],
        "scratchpad": {**state.scratchpad, "job_ids": job_ids},
    }


def pre_done(state: HPCState) -> dict:
    """Return job IDs as artifacts to supervisor."""
    job_ids = state.scratchpad.get("job_ids", {})

    return {
        "status": TaskStatus.DONE,
        "result_data": HPCResult(
            job_ids=job_ids,
            node_assignment={a: "node01" for a in job_ids},
        ),
        "artifacts_out": [
            Artifact(artifact_id=f"{state.task_id}-job-{agent_name}-{job_id}",
                     path="", type="slurm",
                     description=f"HPC job {job_id} for {agent_name}",
                     producer_agent="hpc", producer_task_id=state.task_id)
            for agent_name, job_id in job_ids.items()
        ],
        "node_history": state.node_history + ["pre_done"],
    }


# ═══════════════════════════════════════════════════════════════════

def _finalize(state: HPCState) -> dict:
    return {"agent_result": state.to_agent_result("hpc")}


hpc_graph = (
    StateGraph(HPCState)
    .add_node("query_resources", query_resources)
    .add_node("refine_slurm", refine_slurm)
    .add_node("submit", submit)
    .add_node("pre_done", pre_done)
    .add_node("finalize", _finalize)
    .add_edge(START, "query_resources")
    .add_edge("query_resources", "refine_slurm")
    .add_edge("refine_slurm", "submit")
    .add_edge("submit", "pre_done")
    .add_edge("pre_done", "finalize")
    .add_edge("finalize", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="hpc",
        subgraph=hpc_graph,
        retry_policy=RetryPolicy(
            is_idempotent=False, max_retries=2,
            requires_cleanup_before_retry=True,
            cleanup_steps=["scancel old jobs"],
        ),
        timeout_s=60,
        dependencies=set(),
        description="HPC agent — squeue, refine slurm, sbatch",
    )
