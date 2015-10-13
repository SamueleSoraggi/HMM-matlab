function [Data,nbSamples] = grabDataFromCursorDynamics

nbSamples = 0;
exitFlag = 0;
runningFlag = 0;
nbData = 500;
dt = 0.1;

disp('-Left mouse button to draw several trajectories');
disp('-Right mouse button when done');

fig = figure('position',[10 170 550 550]); hold on; box on;
set(gca,'Xtick',[]); set(gca,'Ytick',[]);

setappdata(gcf,'motion',[]);
setappdata(gcf,'data',[]);
setappdata(gcf,'nbSamples',nbSamples);
setappdata(gcf,'exitFlag',exitFlag);
setappdata(gcf,'runningFlag',runningFlag);

plot(-1,-1);
axis([-0.1 0.1 -0.1 0.1]);
set(fig,'WindowButtonDownFcn',{@wbd});

motion=[];
Data=[];
while(exitFlag==0)
  figure(fig);
  while(runningFlag==1)
    figure(fig);
    cur_point = get(gca,'Currentpoint');
    motion = [motion cur_point(1,1:2)'];
    plot(motion(1,end),motion(2,end),'k.','markerSize',1);
    runningFlag = getappdata(gcf,'runningFlag');
    if(runningFlag==0)
      duration = getappdata(gcf,'duration');
      %Resampling 
      xx = linspace(1,size(motion,2),nbData); 
      motion = spline(1:size(motion,2), motion, xx);
      motion_smooth(1,:) = smooth(motion(1,:),3);
      motion_smooth(2,:) = smooth(motion(2,:),3);
      plot(motion_smooth(1,:),motion_smooth(2,:), 'r', 'lineWidth', 1);
      nbSamples = nbSamples + 1;
      Data = [Data [linspace(0,duration,nbData); motion_smooth]];
      motion=[];
    end
  end
  runningFlag = getappdata(gcf,'runningFlag');
  exitFlag = getappdata(gcf,'exitFlag');
end

for n=1:nbSamples
  %Compute velocity
  Data(4:5,(n-1)*nbData+1:n*nbData) = ([Data(2:3,(n-1)*nbData+2:n*nbData) Data(2:3,n*nbData)] - ...
    Data(2:3,(n-1)*nbData+1:n*nbData)) / dt;
end

close all;


% -----------------------------------------------------------------------
function wbd(h,evd) % executes when the mouse button is pressed
muoseside = get(gcf,'SelectionType');
if strcmp(muoseside,'alt')==1
  setappdata(gcf,'exitFlag',1);
  return;
end
%get the values and store them in the figure's appdata
props.WindowButtonMotionFcn = get(h,'WindowButtonMotionFcn');
props.WindowButtonUpFcn = get(h,'WindowButtonUpFcn');
setappdata(h,'TestGuiCallbacks',props);
set(h,'WindowButtonMotionFcn',{@wbm});
set(h,'WindowButtonUpFcn',{@wbu});
setappdata(gcf,'runningFlag',1);
tic
% -----------------------------------------------------------------------
function wbm(h,evd) % executes while the mouse moves
% -----------------------------------------------------------------------
function wbu(h,evd) % executes when the mouse button is released
setappdata(gcf,'runningFlag',0);
duration = toc;
setappdata(gcf,'duration',duration);
%get the properties and restore them
props = getappdata(h,'TestGuiCallbacks');
set(h,props);
