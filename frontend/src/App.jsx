import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Mic, Send, CloudRain, TrendingUp, Settings, MessageSquare,
  History, User, Leaf, Droplet, Bug, FileText, ArrowRight,
  Trash2, X, Volume2, VolumeX, RefreshCw,
} from 'lucide-react'
import './index.css'

const API_BASE = '/api'

const SUGGESTIONS = [
  {
    icon: Leaf,
    label: 'Beginner Farming Guide',
    query: 'I want to start farming. What are the very first steps I should take regarding soil testing, crop selection, and land preparation?',
    image: 'https://images.unsplash.com/photo-1592982537447-6f2a6a0a3023?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60',
  },
  {
    icon: Droplet,
    label: 'Irrigation & Fertilizer',
    query: 'What are the best practices for irrigation and fertilizer application for crops in my area?',
    image: 'https://images.unsplash.com/photo-1563514227147-6d2ff665a6a0?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60',
  },
  {
    icon: Bug,
    label: 'Identify Pests & Diseases',
    query: 'How do I identify and control common pests and diseases in my crops?',
    image: 'https://images.unsplash.com/photo-1588614407873-1076f8e79878?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60',
  },
  {
    icon: FileText,
    label: 'Gov. Schemes (PM-Kisan)',
    query: 'Tell me about PM-Kisan scheme, eligibility criteria, and documents required to apply.',
    image: 'https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60',
  },
]

const FOLLOW_UPS = [
  { label: '📖 Tell me more', query: 'Can you provide more details on your previous answer regarding this crop or farming technique?' },
  { label: '💰 What about costs?', query: 'What are the estimated costs or financial implications related to this farming method?' },
  { label: '🔄 Any alternatives?', query: 'Are there any alternative farming methods or crop options available?' },
  { label: '📅 Best season?', query: 'What is the best season or time of year for this farming practice?' },
]

const WELCOME_MSG = 'Welcome! I am **KisanAI**, your personal farming assistant. Ask me anything about crops, pests, fertilizers, irrigation, or government schemes — or tap a suggestion below to get started!'

// ── Text-to-Speech via Web Speech API ──────────────────────────────────────
function speakText(text, lang = 'en-US') {
  if (!('speechSynthesis' in window)) return
  window.speechSynthesis.cancel()
  const plain = text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/`(.*?)`/g, '$1')
  const utter = new SpeechSynthesisUtterance(plain)
  utter.lang = lang
  utter.rate = 0.95
  window.speechSynthesis.speak(utter)
}

function App() {
  const [messages, setMessages] = useState([{ role: 'assistant', content: WELCOME_MSG }])
  const [input, setInput] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('chat')
  const [chatHistory, setChatHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('kisanai_history') || '[]') } catch { return [] }
  })
  const [showProfile, setShowProfile] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [profile, setProfile] = useState(() => {
    try { return JSON.parse(localStorage.getItem('kisanai_profile') || '{"name":"","place":"","phone":""}') } catch { return { name: '', place: '', phone: '' } }
  })
  const [profileSaved, setProfileSaved] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(() => localStorage.getItem('kisanai_tts') === 'true')
  const [language, setLanguage] = useState(() => localStorage.getItem('kisanai_lang') || 'English')
  const [voiceError, setVoiceError] = useState('')
  const [weather, setWeather] = useState(null)
  const [prices, setPrices] = useState(null)

  const messagesEndRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)

  // Persist settings
  useEffect(() => { localStorage.setItem('kisanai_tts', ttsEnabled) }, [ttsEnabled])
  useEffect(() => { localStorage.setItem('kisanai_lang', language) }, [language])
  useEffect(() => { localStorage.setItem('kisanai_history', JSON.stringify(chatHistory)) }, [chatHistory])

  // Auto-scroll
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // Fetch weather & prices on mount
  useEffect(() => {
    fetch(`${API_BASE}/weather`)
      .then(r => r.json())
      .then(setWeather)
      .catch(() => {})

    fetch(`${API_BASE}/prices`)
      .then(r => r.json())
      .then(setPrices)
      .catch(() => {})
  }, [])

  // ── Voice recording ────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      setVoiceError('')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop())
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        const formData = new FormData()
        formData.append('file', audioBlob, 'audio.wav')

        try {
          setIsLoading(true)
          const response = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: formData })
          const data = await response.json()
          if (data.text) {
            setInput(data.text)
          } else {
            setVoiceError('No speech detected')
          }
        } catch (err) {
          setVoiceError(`Error: ${err.message}`)
        } finally {
          setIsLoading(false)
        }
      }

      mediaRecorder.start()
      setIsListening(true)
    } catch (err) {
      setVoiceError('Microphone access denied. Please allow microphone permissions.')
      console.error('Microphone error:', err)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop()
      setIsListening(false)
    }
  }

  const handleMicClick = () => {
    if (isListening) stopRecording()
    else startRecording()
  }

  // ── Send message ───────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return
    const userMessage = text.trim()
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setInput('')
    setIsLoading(true)
    setVoiceError('')

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 65000)

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, language }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const data = await response.json()
      const reply = data.response || '⚠️ Empty response from server.'
      setMessages(prev => [...prev, { role: 'assistant', content: reply }])

      if (ttsEnabled) {
        const ttsLang = language === 'Kannada' ? 'kn-IN' : 'en-US'
        speakText(reply, ttsLang)
      }
    } catch (err) {
      clearTimeout(timeoutId)
      const errorMsg = err.name === 'AbortError'
        ? '⏳ The request timed out. The AI is taking too long. Please try again.'
        : '⚠️ Could not reach the backend. Please make sure the FastAPI server is running on port 8000.'
      setMessages(prev => [...prev, { role: 'assistant', content: errorMsg }])
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, language, ttsEnabled])

  // ── Chat history management ────────────────────────────────────────────
  const handleNewChat = () => {
    if (messages.length > 1) {
      const firstUserMsg = messages.find(m => m.role === 'user')
      setChatHistory(prev => [
        {
          id: Date.now(),
          title: (firstUserMsg?.content || 'Chat').slice(0, 50) + '…',
          messages: [...messages],
          date: new Date().toLocaleString(),
        },
        ...prev,
      ])
    }
    setMessages([{ role: 'assistant', content: WELCOME_MSG }])
    setActiveTab('chat')
  }

  const loadChat = (chat) => {
    setMessages(chat.messages)
    setActiveTab('chat')
  }

  const deleteChat = (id, e) => {
    e.stopPropagation()
    setChatHistory(prev => prev.filter(c => c.id !== id))
  }

  const handleSaveProfile = () => {
    localStorage.setItem('kisanai_profile', JSON.stringify(profile))
    setProfileSaved(true)
    setShowProfile(false)
    setTimeout(() => setProfileSaved(false), 3000)
  }

  // ── Markdown renderer ──────────────────────────────────────────────────
  const renderMarkdown = (text) => {
    const html = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-size:0.9em">$1</code>')
      .replace(/\n/g, '<br/>')
    return <span dangerouslySetInnerHTML={{ __html: html }} />
  }

  // ── Derived weather/price display values ──────────────────────────────
  const weatherDisplay = weather || { temperature: '28°C', description: 'Partly Cloudy', humidity: '65%', advice: 'Optimal condition for foliar sprays. No rain expected in the next 24 hours.' }
  const pricesDisplay = prices || [
    { crop: 'Tomato', mandi: 'Mysuru Mandi', price: '₹24/kg', trend: '+2.5%' },
    { crop: 'Onion', mandi: 'Bengaluru', price: '₹45/kg', trend: '-1.2%' },
    { crop: 'Paddy (Rice)', mandi: 'Mandya', price: '₹32/kg', trend: '+0.5%' },
  ]

  return (
    <div className="app-container">
      {/* ── Sidebar ── */}
      <nav className="glass-panel sidebar">
        <div className="logo-container">
          <Leaf size={28} />
        </div>

        <div
          className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
          title="Chat"
          onClick={() => { setActiveTab('chat'); setShowProfile(false); setShowSettings(false) }}
        >
          <MessageSquare size={24} />
        </div>

        <div
          className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
          title="Chat History"
          onClick={() => { setActiveTab('history'); setShowProfile(false); setShowSettings(false) }}
        >
          <History size={24} />
        </div>

        <div
          className={`nav-item ${showProfile ? 'active' : ''}`}
          title="Profile"
          onClick={() => { setShowProfile(p => !p); setShowSettings(false); setActiveTab('chat') }}
        >
          <User size={24} />
        </div>

        <div
          className={`nav-item ${showSettings ? 'active' : ''}`}
          style={{ marginTop: 'auto' }}
          title="Settings"
          onClick={() => { setShowSettings(s => !s); setShowProfile(false); setActiveTab('chat') }}
        >
          <Settings size={24} />
        </div>
      </nav>

      {/* ── Main Content ── */}
      <main className="glass-panel main-chat">
        <header className="chat-header">
          <h1>KisanAI</h1>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button className="header-btn" onClick={handleNewChat} title="New Chat">
              <RefreshCw size={18} /> New Chat
            </button>
            <div className="status-badge">
              <div className="status-dot"></div>
              Online
            </div>
          </div>
        </header>

        {/* Profile Modal */}
        {showProfile && (
          <div className="modal-overlay" onClick={() => setShowProfile(false)}>
            <div className="modal glass-panel" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <h2>👤 Farmer Profile</h2>
                <button className="modal-close" onClick={() => setShowProfile(false)}><X size={20} /></button>
              </div>
              <div className="modal-body">
                <label>Name</label>
                <input
                  type="text"
                  placeholder="Enter your name"
                  value={profile.name}
                  onChange={e => setProfile({ ...profile, name: e.target.value })}
                />
                <label>Location / Village</label>
                <input
                  type="text"
                  placeholder="Enter your place"
                  value={profile.place}
                  onChange={e => setProfile({ ...profile, place: e.target.value })}
                />
                <label>Phone Number</label>
                <input
                  type="text"
                  placeholder="Enter phone number"
                  value={profile.phone}
                  onChange={e => setProfile({ ...profile, phone: e.target.value })}
                />
                <button className="save-btn" onClick={handleSaveProfile}>Save Profile</button>
              </div>
            </div>
          </div>
        )}

        {/* Settings Modal */}
        {showSettings && (
          <div className="modal-overlay" onClick={() => setShowSettings(false)}>
            <div className="modal glass-panel" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <h2>⚙️ Settings</h2>
                <button className="modal-close" onClick={() => setShowSettings(false)}><X size={20} /></button>
              </div>
              <div className="modal-body">
                <div className="setting-row">
                  <span>🔊 Text-to-Speech</span>
                  <button
                    className={`toggle-btn ${ttsEnabled ? 'on' : ''}`}
                    onClick={() => setTtsEnabled(v => !v)}
                  >
                    {ttsEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
                    {ttsEnabled ? 'On' : 'Off'}
                  </button>
                </div>
                <div className="setting-row">
                  <span>🌐 Language</span>
                  <select
                    className="setting-select"
                    value={language}
                    onChange={e => setLanguage(e.target.value)}
                  >
                    <option value="English">English</option>
                    <option value="Kannada">Kannada</option>
                  </select>
                </div>
                {profile.name && (
                  <div className="setting-row">
                    <span>👤 Logged in as</span>
                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{profile.name}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Profile saved toast */}
        {profileSaved && <div className="toast">✅ Profile saved successfully!</div>}

        {/* ── Chat View ── */}
        {activeTab === 'chat' && (
          <>
            <div className="chat-messages">
              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === 'assistant' ? (
                      <img src="https://images.unsplash.com/photo-1534269222346-5a896154c41d?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&q=80" alt="AI" />
                    ) : (
                      <img src="https://images.unsplash.com/photo-1560250097-0b93528c311a?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&q=80" alt="You" />
                    )}
                  </div>
                  <div className={`message ${msg.role}`}>
                    {renderMarkdown(msg.content)}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="message-row assistant">
                  <div className="avatar">
                    <img src="https://images.unsplash.com/photo-1534269222346-5a896154c41d?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&q=80" alt="AI" />
                  </div>
                  <div className="message assistant">
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}

              {/* Suggestion cards — shown only on first load */}
              {messages.length === 1 && !isLoading && (
                <div className="suggestion-grid">
                  {SUGGESTIONS.map((s, i) => (
                    <div
                      key={i}
                      className="suggestion-card"
                      style={{ backgroundImage: `url(${s.image})` }}
                      onClick={() => sendMessage(s.query)}
                    >
                      <div className="suggestion-content">
                        <div className="suggestion-icon"><s.icon size={24} /></div>
                        <div className="suggestion-text">{s.label}</div>
                        <ArrowRight size={20} color="white" style={{ marginLeft: 'auto' }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Follow-up chips — shown after conversation starts */}
              {messages.length > 1 && !isLoading && (
                <div className="followup-chips">
                  {FOLLOW_UPS.map((f, i) => (
                    <button key={i} className="followup-chip" onClick={() => sendMessage(f.query)}>
                      {f.label}
                    </button>
                  ))}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
              <button
                className={`mic-btn ${isListening ? 'active' : ''}`}
                onClick={handleMicClick}
                disabled={isLoading}
                title={isListening ? '🔴 Stop recording' : '🎤 Click to record'}
              >
                <Mic size={28} />
              </button>

              {isListening && (
                <div style={{ color: '#ef4444', fontSize: '0.85rem', marginLeft: '8px', flex: 1, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  🔴 <span style={{ animation: 'pulse 1.5s infinite' }}>Recording…</span>
                </div>
              )}

              {voiceError && !isListening && (
                <div style={{ color: '#ef4444', fontSize: '0.85rem', marginLeft: '8px', flex: 1 }}>
                  ⚠️ {voiceError}
                </div>
              )}

              <div className="input-wrapper">
                <input
                  type="text"
                  className="chat-input"
                  placeholder={language === 'Kannada' ? 'ಕೃಷಿ ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ...' : 'Ask anything about farming...'}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage(input)}
                  disabled={isLoading}
                />
                <button
                  className="send-btn"
                  onClick={() => sendMessage(input)}
                  disabled={isLoading || !input.trim()}
                  title="Send"
                >
                  <Send size={22} />
                </button>
              </div>
            </div>
          </>
        )}

        {/* ── History View ── */}
        {activeTab === 'history' && (
          <div className="history-view">
            <h2 style={{ padding: '24px 32px', borderBottom: '1px solid var(--border-light)' }}>📋 Chat History</h2>
            <div className="history-list">
              {chatHistory.length === 0 ? (
                <div className="empty-state">
                  <History size={48} color="var(--text-muted)" />
                  <p>No saved conversations yet.</p>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                    When you start a new chat, the previous one is automatically saved here.
                  </p>
                </div>
              ) : (
                chatHistory.map(chat => (
                  <div key={chat.id} className="history-item" onClick={() => loadChat(chat)}>
                    <div>
                      <div className="history-title">{chat.title}</div>
                      <div className="history-date">{chat.date}</div>
                    </div>
                    <button
                      className="delete-btn"
                      onClick={e => deleteChat(chat.id, e)}
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </main>

      {/* ── Right Panel ── */}
      <aside className="right-panel">
        <div className="glass-panel widget">
          <h3><CloudRain size={20} color="var(--accent-green)" /> Local Weather</h3>
          <div className="weather-info">
            <div className="weather-temp">{weatherDisplay.temperature}</div>
            <div>
              <div className="weather-desc">{weatherDisplay.description}</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Humidity: {weatherDisplay.humidity}
              </div>
            </div>
          </div>
          <p style={{ fontSize: '0.95rem', marginTop: '12px', color: '#cbd5e1', lineHeight: '1.5' }}>
            {weatherDisplay.advice}
          </p>
        </div>

        <div className="glass-panel widget">
          <h3><TrendingUp size={20} color="var(--accent-green)" /> Market Prices</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '12px' }}>
            {pricesDisplay.map((item, i) => {
              const isUp = (item.trend || '').startsWith('+')
              return (
                <div key={i} className="mandi-item">
                  <div>
                    <strong style={{ fontSize: '1.05rem' }}>{item.crop}</strong>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{item.mandi}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: isUp ? 'var(--accent-green)' : '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>
                      {item.price}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: isUp ? 'var(--accent-green)' : '#ef4444' }}>
                      {item.trend}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </aside>
    </div>
  )
}

export default App
