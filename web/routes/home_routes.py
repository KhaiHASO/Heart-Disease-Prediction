from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "123456":
            return f"Chào {username}"
        else:
            message = "Sai thông tin đăng nhập"
    return render_template("index.html", message=message)
