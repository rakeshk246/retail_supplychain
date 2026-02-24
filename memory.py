# memory.py
# ==============================================================================
# Phase 3: Vector Memory for Agent Learning
# ChromaDB-based episodic memory — agents remember past decisions & outcomes
# ==============================================================================

import json
import time
from datetime import datetime
from typing import List, Dict, Optional

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class AgentMemory:
    """Vector memory store for supply chain agents.
    
    Each agent stores 'episodes' — past decisions and their outcomes.
    When making new decisions, agents retrieve similar past situations
    to inform their reasoning (few-shot learning from experience).
    
    Uses ChromaDB for vector similarity search. Falls back to a simple
    list-based memory if ChromaDB is unavailable.
    """

    def __init__(self, agent_name: str, persist_dir: str = None):
        self.agent_name = agent_name
        self.episodes = []  # Fallback in-memory store
        self.collection = None
        self._initialized = False

        if HAS_CHROMADB:
            try:
                if persist_dir:
                    self.client = chromadb.PersistentClient(path=persist_dir)
                else:
                    self.client = chromadb.Client()  # In-memory

                # Each agent gets its own collection
                self.collection = self.client.get_or_create_collection(
                    name=f"agent_{agent_name.lower().replace(' ', '_')}",
                    metadata={"agent": agent_name}
                )
                self._initialized = True
            except Exception as e:
                print(f"ChromaDB init failed for {agent_name}: {e}. Using fallback memory.")
        else:
            print(f"ChromaDB not installed. Agent {agent_name} using in-memory fallback.")

    @property
    def is_vector_store(self):
        return self._initialized and self.collection is not None

    def store_episode(self, situation: str, decision: str, outcome: dict,
                      day: int = 0, metadata: dict = None):
        """Store a decision episode in memory.
        
        Args:
            situation: description of the situation/context
            decision: what the agent decided
            outcome: result of the decision (e.g., {'success': True, 'fill_rate': 95})
            day: simulation day
            metadata: additional metadata
        """
        episode = {
            'situation': situation,
            'decision': decision,
            'outcome': outcome,
            'day': day,
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name
        }
        if metadata:
            episode.update(metadata)

        # Store in vector DB
        if self.is_vector_store:
            try:
                doc_text = f"Situation: {situation}\nDecision: {decision}\nOutcome: {json.dumps(outcome)}"
                doc_id = f"{self.agent_name}_{day}_{int(time.time() * 1000)}"

                self.collection.add(
                    documents=[doc_text],
                    ids=[doc_id],
                    metadatas=[{
                        'day': str(day),
                        'success': str(outcome.get('success', True)),
                        'agent': self.agent_name,
                        'decision_type': metadata.get('decision_type', 'general') if metadata else 'general'
                    }]
                )
            except Exception as e:
                print(f"Vector store error: {e}")

        # Always store in fallback list too
        self.episodes.append(episode)

    def recall_similar(self, current_situation: str, n_results: int = 3) -> List[Dict]:
        """Retrieve similar past episodes using vector similarity.
        
        Args:
            current_situation: description of the current situation
            n_results: number of similar episodes to return
            
        Returns:
            List of similar past episodes
        """
        if self.is_vector_store and self.collection.count() > 0:
            try:
                results = self.collection.query(
                    query_texts=[current_situation],
                    n_results=min(n_results, self.collection.count())
                )

                episodes = []
                if results and results['documents']:
                    for doc, meta in zip(results['documents'][0],
                                         results['metadatas'][0]):
                        episodes.append({
                            'document': doc,
                            'metadata': meta
                        })
                return episodes
            except Exception as e:
                print(f"Vector recall error: {e}")

        # Fallback: return most recent episodes
        return self.episodes[-n_results:] if self.episodes else []

    def recall_by_type(self, decision_type: str, n_results: int = 5) -> List[Dict]:
        """Retrieve episodes filtered by decision type."""
        if self.is_vector_store:
            try:
                results = self.collection.get(
                    where={"decision_type": decision_type},
                    limit=n_results
                )
                if results and results['documents']:
                    return [
                        {'document': doc, 'metadata': meta}
                        for doc, meta in zip(results['documents'], results['metadatas'])
                    ]
            except Exception:
                pass

        # Fallback
        filtered = [e for e in self.episodes
                    if e.get('decision_type') == decision_type]
        return filtered[-n_results:]

    def get_success_rate(self) -> float:
        """Calculate success rate from stored episodes."""
        if not self.episodes:
            return 0.0
        successes = sum(1 for e in self.episodes
                        if e.get('outcome', {}).get('success', False))
        return successes / len(self.episodes)

    def get_episode_count(self) -> int:
        """Total number of stored episodes."""
        if self.is_vector_store:
            return self.collection.count()
        return len(self.episodes)

    def format_for_prompt(self, episodes: List[Dict], max_episodes: int = 3) -> str:
        """Format retrieved episodes into text for LLM prompt injection.
        
        This is the key connection between memory and ReAct reasoning:
        past episodes become few-shot examples in the agent's prompt.
        """
        if not episodes:
            return "No relevant past experience found."

        lines = ["Relevant past experiences:"]
        for i, ep in enumerate(episodes[:max_episodes], 1):
            if isinstance(ep, dict) and 'document' in ep:
                lines.append(f"\n--- Experience {i} ---")
                lines.append(ep['document'])
                if 'metadata' in ep:
                    lines.append(f"Success: {ep['metadata'].get('success', 'unknown')}")
            elif isinstance(ep, dict) and 'situation' in ep:
                lines.append(f"\n--- Experience {i} (Day {ep.get('day', '?')}) ---")
                lines.append(f"Situation: {ep['situation']}")
                lines.append(f"Decision: {ep['decision']}")
                lines.append(f"Outcome: {json.dumps(ep.get('outcome', {}))}")

        return "\n".join(lines)

    def clear(self):
        """Clear all memory."""
        self.episodes = []
        if self.is_vector_store:
            try:
                # Delete and recreate collection
                self.client.delete_collection(
                    f"agent_{self.agent_name.lower().replace(' ', '_')}"
                )
                self.collection = self.client.get_or_create_collection(
                    name=f"agent_{self.agent_name.lower().replace(' ', '_')}",
                    metadata={"agent": self.agent_name}
                )
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'agent': self.agent_name,
            'total_episodes': self.get_episode_count(),
            'success_rate': round(self.get_success_rate(), 2),
            'using_vector_store': self.is_vector_store
        }
