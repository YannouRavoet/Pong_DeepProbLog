nn(pong_net_y,[Img],BallY,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) :: y_coord(Img,BallY).
nn(pong_net_x,[Img],BallX,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]) :: x_coord(Img,BallX).

%used to train the pong_net_x network
distance_x(Img, D) :- x_coord(Img, BallX), D is 29 - BallX.
coordinates(Img, X, Y):- x_coord(Img,X), y_coord(Img, Y).

choose_action(PrevImg, CurImg, AIY, Action):-
    coordinates(PrevImg, BallPrevX, BallPrevY),
    coordinates(CurImg, BallCurX, BallCurY),
    choose_action(AIY, BallPrevX, BallPrevY, BallCurX, BallCurY, Action).

%going away from agent
choose_action(AIY, BallPrevX, _, BallCurX, _, noop):-
    direction_x(BallPrevX, BallCurX, -1),
    AIY = 8.
choose_action(AIY, BallPrevX, _, BallCurX, _, up):-
    direction_x(BallPrevX, BallCurX, -1),
    AIY > 8.
choose_action(AIY, BallPrevX, _, BallCurX, _, down):-
    direction_x(BallPrevX, BallCurX, -1),
    AIY < 8.

%going towards agent
choose_action(AIY, BallPrevX, BallPrevY, BallCurX, BallCurY, noop):-
    direction_x(BallPrevX, BallCurX, 1), %coming towards AI
    intersect_y(BallPrevX, BallPrevY, BallCurX, BallCurY, Y),
    AIY = Y.
choose_action(AIY, BallPrevX, BallPrevY, BallCurX, BallCurY, up):-
    direction_x(BallPrevX, BallCurX, 1), %coming towards AI
    intersect_y(BallPrevX, BallPrevY, BallCurX, BallCurY, Y),
    AIY > Y.
choose_action(AIY, BallPrevX, BallPrevY, BallCurX, BallCurY, down):-
    direction_x(BallPrevX, BallCurX, 1), %coming towards AI
    intersect_y(BallPrevX, BallPrevY, BallCurX, BallCurY, Y),
    AIY < Y.

direction_x(PrevX, CurX, 1):- PrevX =< CurX.
direction_x(PrevX, CurX, -1):- PrevX > CurX.

speed(Cur, Cur, 1).%hack to deal with bounces = default to react
speed(Prev, Cur, Speed):-
    (Cur > Prev; Cur < Prev),
    Speed is Cur - Prev.

intersect_y(BallPrevX, BallPrevY, BallCurX, BallCurY, Y):-
    speed(BallPrevX, BallCurX, SpeedX),
    speed(BallPrevY, BallCurY, SpeedY),
    DistX is 29 - BallCurX,
    FramesLeft is DistX / SpeedX,
    intersect_y_iter(SpeedY, FramesLeft, BallCurY, Y).

intersect_y_iter(_, FramesLeft, Y, Y):-
    FramesLeft =< 0.
intersect_y_iter(SpeedY, FramesLeft, TempY, Y):-
    FramesLeft > 0,
    NewFramesLeft is FramesLeft - 1,
    bounded_y_calc(TempY, SpeedY, NewTempY, NewSpeedY),
    intersect_y_iter(NewSpeedY, NewFramesLeft, NewTempY, Y).

%calculates the next y position and updates yspeed if there was a bounce
bounded_y_calc(CurY, SpeedY, NewY, NewSpeedY):-
    Steps is abs(SpeedY),
    DirY is sign(SpeedY),
    bounded_y_calc_iter(CurY, DirY, NewY, NewDirY, Steps),
    NewSpeedY is Steps * NewDirY.

bounded_y_calc_iter(CurY, DirY, CurY, DirY, 0).
bounded_y_calc_iter(CurY, DirY, FinalY, FinalDirY, Steps):-
    Steps > 0,
    (CurY = 0; CurY = 15), %if next step would hit top or bottom of screen
    NewDirY is -1*DirY,
    TargetY is CurY + NewDirY,
    NewSteps is Steps - 1,
    bounded_y_calc_iter(TargetY, NewDirY, FinalY, FinalDirY, NewSteps).
bounded_y_calc_iter(CurY, DirY, FinalY, FinalDirY, Steps):-
    Steps > 0,
    CurY > 0,
    CurY < 15,
    TargetY is CurY + DirY,
    NewSteps is Steps - 1,
    bounded_y_calc_iter(TargetY, DirY, FinalY, FinalDirY, NewSteps).