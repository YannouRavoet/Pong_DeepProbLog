nn(pong_net_y,[Img],BallY,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) :: y_coord(Img,BallY).

distance_y(Img, AIY, D):- y_coord(Img, BallY), D is AIY - BallY.

choose_action(Img, AIY, noop) :- distance_y(Img, AIY, D), D = 0.
choose_action(Img, AIY, up) :- distance_y(Img, AIY, D), D > 0.
choose_action(Img, AIY, down) :- distance_y(Img, AIY, D), D < 0.