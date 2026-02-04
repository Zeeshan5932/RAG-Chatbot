from langchain.memory import ConversationBufferMemory


class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "qa_chain": None,
                "memory": ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                ),
            }
        return self.sessions[session_id]