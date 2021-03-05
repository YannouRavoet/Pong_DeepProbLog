nn(pong_net,[Img],BallY,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) :: y_coord(Img,BallY).

choose_action(PaddleY, Img, noop) :- y_coord(Img, BallY), PaddleY = BallY.
choose_action(PaddleY, Img, up) :- y_coord(Img, BallY), PaddleY > BallY.
choose_action(PaddleY, Img, down) :- y_coord(Img, BallY), PaddleY < BallY.