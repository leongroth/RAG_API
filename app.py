from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from rag import create_vectorstore, get_answer



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://experiencepathways.com"]}})


# Init vectorstore from source.txt
@app.route("/api/hello", methods=["GET", "POST"])
def init_vectorstore():
    try:
        print("🔥 /api/hello called")
        create_vectorstore()
        return jsonify({"status": "Vectorstore created ✅"})
    except Exception as e:
        print("❌ Error creating vectorstore:", e)
        return jsonify({"error": str(e)}), 500

# Generate answer from user question
@app.route("/api/generate", methods=["POST"])
def generate_answer():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        print("💬 /api/generate called with:", question)
        answer = get_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Error generating answer:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
