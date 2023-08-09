from flask import Flask

  
app = Flask(__name__)
  
def show_user(username):
    # Greet the user asjdasjldksajldlsajdjadppoiapdipadjlkadjlk
    return f'Hello {username} !'
    
app.add_url_rule('/user/<username>', 'show_user', show_user)
  
if __name__ == "_main_":
    app.run(debug=True)


    