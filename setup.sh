mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"arghyadeep.d@somaiya.edu\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
