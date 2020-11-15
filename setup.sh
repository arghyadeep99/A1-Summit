mkdir -p ~/.streamlit/

echo "\
	[general]\n\
	email=\"arghyadeep.d@somaiya.edu\"\n\
	" > ~/.streamlit/credentials.toml

echo "\
	[server]\n\
	headless = true\n\
	port = $PORT\n\
	enableCORS = false\n\
	\n\
	" > ~/.streamlit/config.toml
