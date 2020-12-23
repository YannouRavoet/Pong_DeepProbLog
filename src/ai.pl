nn(pong_net,[Y],P,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) :: y_coord(Y,P).


choose_action(BallY, BallY, 0) :- y_coord(BallY, Prob).
choose_action(PaddleY, BallY, -1) :- y_coord(BallY, Prob), PaddleY > BallY.
choose_action(PaddleY, BallY, 1) :- y_coord(BallY, Prob), PaddleY < BallY.