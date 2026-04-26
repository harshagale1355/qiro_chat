import { useState, useEffect, useRef } from 'react';
import './index.css';

const API_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  
  const chatWindowRef = useRef(null);
  
  // Web Speech API Initialization
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  if (recognition) {
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      handleSend(transcript);
    };

    recognition.onspeechend = () => {
      recognition.stop();
      setIsListening(false);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error', event.error);
      setIsListening(false);
    };
  }

  const toggleListen = () => {
    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      if (recognition) {
        recognition.start();
        setIsListening(true);
      } else {
        alert("Your browser does not support Speech Recognition.");
      }
    }
  };

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.1;
      window.speechSynthesis.speak(utterance);
    }
  };

  const scrollToBottom = () => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history?session_id=default_user`);
      const data = await res.json();
      if (data.history && data.history.length > 0) {
        setMessages(data.history);
      } else {
        setMessages([{ role: 'assistant', content: 'Welcome to Qiro Verse! How can I assist you today?' }]);
      }
    } catch (e) {
      console.error("Error fetching history", e);
      setMessages([{ role: 'assistant', content: 'Welcome to Qiro Verse! How can I assist you today?' }]);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSend = async (messageOverride = null) => {
    const text = messageOverride || input;
    if (!text.trim()) return;

    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: text, session_id: "default_user" })
      });
      
      const data = await res.json();
      const assistantMessage = { role: 'assistant', content: data.response };
      
      setMessages(prev => [...prev, assistantMessage]);
      speakText(data.response);

    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Is the server running?' }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="app-container">
        <header className="chat-header">
          <div className="chat-title">Qiro Verse</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Online</div>
        </header>

        <main className="chat-window" ref={chatWindowRef}>
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.role === 'assistant' ? (
                // Use dangerouslySetInnerHTML or simple string for bold logic if needed
                <span dangerouslySetInnerHTML={{ __html: msg.content.replace(/\n/g, '<br/>') }} />
              ) : (
                msg.content
              )}
            </div>
          ))}
          {isTyping && <div className="typing-indicator">Qiro is thinking...</div>}
        </main>

        <div className="chat-input-container">
          <button 
            className={`btn btn-mic ${isListening ? 'mic-active' : ''}`}
            onClick={toggleListen}
            title="🎤 Voice Input"
          >
            {isListening ? '🎙️' : '🎤'}
          </button>
          
          <input 
            type="text" 
            placeholder="Type your message..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          
          <button className="btn btn-send" onClick={() => handleSend()}>
            ➤
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
