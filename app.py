from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder='/web/templates')

# Variable global para almacenar la conversación (no recomendado para producción)
conversation = []

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            # Generar una respuesta placeholder para el bot
            bot_response = f"Bot response (placeholder) for: {user_input}"
            conversation.append({"sender": "User", "message": user_input})
            conversation.append({"sender": "Bot", "message": bot_response})
        return redirect(url_for("index"))
    return render_template("index.html", conversation=conversation)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
