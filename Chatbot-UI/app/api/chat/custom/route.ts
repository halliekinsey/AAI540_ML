import { Database } from "@/supabase/types"
import { ChatSettings } from "@/types"
import { createClient } from "@supabase/supabase-js"
import { OpenAIStream, StreamingTextResponse } from "ai"
import { ServerRuntime } from "next"
import OpenAI from "openai"
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions.mjs"

export const runtime = "edge";

export async function POST(request: Request) {
  try {
    const json = await request.json();
    const { messages } = json as { messages: any[] };

    // Extract user's last message
    const userMessage = messages?.[messages.length - 1]?.content || "";
    if (!userMessage) {
      return new Response(
        JSON.stringify({ message: "No message provided." }),
        { status: 400 }
      );
    }

    console.log("🔄 Sending request to FastAPI backend...");

    // Call FastAPI backend (queries Pinecone + LM Studio)
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage }),
    });

    if (!response.ok) {
      throw new Error(`FastAPI responded with status ${response.status}`);
    }

    const data = await response.json();
    console.log("✅ Response from FastAPI:", data);

    // Return the response to the frontend
    return new Response(JSON.stringify({ response: data.response }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  } catch (error: any) {
    console.error("❌ Error in API route:", error);
    return new Response(
      JSON.stringify({ message: "Server error: " + error.message }),
      { status: 500 }
    );
  }
}

