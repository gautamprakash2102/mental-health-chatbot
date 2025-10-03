import React, { useState } from "react";
import axios from "axios";

interface Message {
  sender: "user" | "bot";
  text: string;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");

  const BACKEND_URL = "http://127.0.0.1:8000/chat";

    const sendMessage = async () => {
    if (!input.trim()) return;

    // add user message
    setMessages((prev) => [...prev, { sender: "user", text: input }]);

    try {
        const res = await axios.get(BACKEND_URL, {
        params: { query: input },
        });

        const botReply = res.data.answer || "No response.";
        setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
        setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error contacting backend." },
        ]);
    }

    // ✅ clear input after sending
    setInput("");
    };


  return (
    <div className="flex flex-col h-screen p-4 bg-gray-100">
      <div className="flex-1 overflow-y-auto space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-2 rounded-lg max-w-md ${
              msg.sender === "user"
                ? "bg-blue-500 text-white self-end ml-auto"
                : "bg-gray-300 text-black self-start mr-auto"
            }`}
          >
            {msg.text}
            {msg.sender === "bot" && (
              <div className="flex gap-2 mt-1">
                <button className="text-green-600">👍</button>
                <button className="text-red-600">👎</button>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex mt-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          className="flex-1 p-2 border rounded-lg"
          placeholder="Type your message..."
        />
        <button
          onClick={sendMessage}
          className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chat;
