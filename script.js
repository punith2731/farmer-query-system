const { useEffect, useMemo, useRef, useState } = React;

const initialConversations = [
  {
    id: "chat-1",
    title: "Welcome Chat",
    messages: [
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "# Welcome 👋\nI am your **AI Assistant**.\n\nTry asking:\n- Explain crop rotation\n- Give me irrigation tips for tomato\n- Show a Python example\n\n```python\ndef greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n```",
      },
    ],
  },
  {
    id: "chat-2",
    title: "Fertilizer Guide",
    messages: [
      {
        id: crypto.randomUUID(),
        role: "user",
        content: "Can you give me a basic NPK schedule for paddy?",
      },
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "Sure. A general split approach:\n1. Basal dose before transplanting\n2. First top dressing at tillering\n3. Second top dressing at panicle initiation\n\nAdjust based on soil test for best results.",
      },
    ],
  },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

function renderMarkdown(markdown) {
  const rawHtml = marked.parse(markdown ?? "", {
    breaks: true,
    gfm: true,
  });
  return DOMPurify.sanitize(rawHtml);
}

function TypingIndicator() {
  return (
    <div className="mb-4 flex items-start justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-tl-sm border border-white bg-black px-4 py-3 text-white">
        <div className="flex items-center gap-1">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const html = useMemo(() => renderMarkdown(message.content), [message.content]);
  const bubbleRef = useRef(null);

  useEffect(() => {
    if (!bubbleRef.current) return;

    const codeBlocks = bubbleRef.current.querySelectorAll("pre code");
    codeBlocks.forEach((codeBlock) => {
      const pre = codeBlock.parentElement;
      if (!pre || pre.dataset.enhanced === "true") return;

      pre.dataset.enhanced = "true";

      const wrapper = document.createElement("div");
      wrapper.className = "code-wrapper";
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      const copyButton = document.createElement("button");
      copyButton.className = "copy-code-btn";
      copyButton.type = "button";
      copyButton.textContent = "Copy";
      copyButton.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(codeBlock.textContent || "");
          copyButton.textContent = "Copied!";
          setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        } catch {
          copyButton.textContent = "Failed";
          setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        }
      });

      wrapper.appendChild(copyButton);
    });
  }, [html]);

  return (
    <div
      className={classNames(
        "mb-4 flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={classNames(
          "markdown-body max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-7",
          isUser
            ? "rounded-tr-sm border border-white bg-white text-black"
            : "rounded-tl-sm bg-black text-white"
        )}
        ref={bubbleRef}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

function Sidebar({ conversations, activeId, onSelect, onNewChat, isOpen, onClose }) {
  return (
    <>
      <div
        className={classNames(
          "fixed inset-0 z-20 bg-black/80 lg:hidden",
          isOpen ? "opacity-100" : "pointer-events-none opacity-0"
        )}
        onClick={onClose}
      />

      <aside
        className={classNames(
          "fixed inset-y-0 left-0 z-30 w-72 border-r border-white bg-black p-3 lg:static lg:z-0 lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <button
          className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl border border-white bg-black px-4 py-3 text-sm font-medium text-white hover:bg-white hover:text-black"
          onClick={onNewChat}
        >
          <span className="text-base">＋</span>
          New Chat
        </button>

        <div className="chat-scroll h-[calc(100vh-92px)] space-y-2 overflow-y-auto pr-1">
          {conversations.map((chat) => (
            <button
              key={chat.id}
              onClick={() => {
                onSelect(chat.id);
                onClose();
              }}
              className={classNames(
                "w-full rounded-xl px-3 py-2 text-left text-sm border",
                chat.id === activeId
                  ? "border-white bg-white text-black"
                  : "border-white bg-black text-white hover:bg-white hover:text-black"
              )}
            >
              <p className="truncate">{chat.title}</p>
            </button>
          ))}
        </div>
      </aside>
    </>
  );
}

function Header({ onOpenSidebar }) {
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-white bg-black px-4">
      <div className="flex items-center gap-2">
        <button
          className="rounded-lg border border-white bg-black p-2 text-white hover:bg-white hover:text-black lg:hidden"
          onClick={onOpenSidebar}
          aria-label="Open chat history"
        >
          ☰
        </button>
        <h1 className="text-sm font-semibold text-white">AI Assistant</h1>
      </div>
      <button
        className="rounded-lg border border-white bg-black px-3 py-1.5 text-xs text-white hover:bg-white hover:text-black"
        aria-label="Settings"
      >
        ⚙
      </button>
    </header>
  );
}

function InputBar({ value, onChange, onSend, disabled, onFilePick }) {
  const fileInputRef = useRef(null);

  return (
    <div className="sticky bottom-0 border-t border-white bg-black px-3 pb-3 pt-2">
      <div className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-2xl border border-white bg-black p-2">
        <button
          className="rounded-lg border border-transparent p-2 text-white hover:border-white"
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Attach file"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={(e) => onFilePick(e.target.files?.[0])}
        />

        <textarea
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          placeholder="Message AI Assistant..."
          className="chat-input max-h-36 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-sm text-white placeholder:text-white/60 focus:outline-none"
        />

        <button
          className="rounded-xl border border-white bg-white p-2.5 text-black hover:bg-black hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
          disabled={disabled}
          onClick={onSend}
          aria-label="Send message"
          title="Send"
        >
          ➤
        </button>
      </div>
    </div>
  );
}

function App() {
  const [conversations, setConversations] = useState(initialConversations);
  const [activeId, setActiveId] = useState(initialConversations[0].id);
  const [draft, setDraft] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const activeConversation = useMemo(
    () => conversations.find((chat) => chat.id === activeId),
    [conversations, activeId]
  );

  const listRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    if (!endRef.current) return;
    endRef.current.scrollIntoView({ block: "end" });
  }, [activeConversation?.messages?.length, isTyping]);

  function createNewChat() {
    const newChat = {
      id: `chat-${Date.now()}`,
      title: "New Conversation",
      messages: [
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "New chat created. Ask me anything.",
        },
      ],
    };

    setConversations((prev) => [newChat, ...prev]);
    setActiveId(newChat.id);
    setSidebarOpen(false);
  }

  function updateConversation(conversationId, updater) {
    setConversations((prev) =>
      prev.map((chat) => (chat.id === conversationId ? updater(chat) : chat))
    );
  }

  function fakeAssistantResponse(userText) {
    const prompt = userText.trim();
    if (!prompt) {
      return "Could you provide a bit more detail?";
    }
    return (
      `You asked: **${prompt}**\n\n` +
      "Here is a concise response with markdown support:\n" +
      "- Clean UI\n- Fast interaction\n- Code friendly formatting\n\n" +
      "```javascript\n" +
      "const nextStep = 'Connect this UI to your backend API';\n" +
      "console.log(nextStep);\n" +
      "```"
    );
  }

  async function handleSend() {
    const text = draft.trim();
    if (!text || !activeConversation || isTyping) return;

    setDraft("");

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };

    updateConversation(activeConversation.id, (chat) => ({
      ...chat,
      title: chat.title === "New Conversation" ? text.slice(0, 26) : chat.title,
      messages: [...chat.messages, userMessage],
    }));

    setIsTyping(true);
    await new Promise((resolve) => setTimeout(resolve, 950));

    const assistantMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: fakeAssistantResponse(text),
    };

    updateConversation(activeConversation.id, (chat) => ({
      ...chat,
      messages: [...chat.messages, assistantMessage],
    }));

    setIsTyping(false);
  }

  function handleFilePick(file) {
    if (!file || !activeConversation) return;
    const fileNote = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: `Attached file: **${file.name}** (${Math.max(1, Math.round(file.size / 1024))} KB)`
    };

    updateConversation(activeConversation.id, (chat) => ({
      ...chat,
      messages: [...chat.messages, fileNote],
    }));
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-black text-white">
      <div className="grid h-full grid-cols-1 lg:grid-cols-[18rem_1fr]">
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={setActiveId}
          onNewChat={createNewChat}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <main className="flex min-h-0 flex-col">
          <Header onOpenSidebar={() => setSidebarOpen(true)} />

          <section className="relative flex min-h-0 flex-1 flex-col">
            <div
              ref={listRef}
              className="chat-scroll mx-auto w-full max-w-3xl flex-1 overflow-y-auto px-3 py-5 md:px-5"
            >
              {(activeConversation?.messages || []).map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {isTyping && <TypingIndicator />}
              <div ref={endRef} />
            </div>

            <InputBar
              value={draft}
              onChange={setDraft}
              onSend={handleSend}
              disabled={isTyping || !draft.trim()}
              onFilePick={handleFilePick}
            />
          </section>
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
