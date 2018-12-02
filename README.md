# ShutterMate ResNeXT+Stockfish Server

https://hub.docker.com/r/alexpetrusca/shuttermate/

# How to Run
* build docker image and run as docker container
* run main python executable

# Server Endpoints

* /digitize GET
  * returns fen code for input birds-eye-view image of a chess board
  
* /nextMove GET
  * returns next best move for input fen code
