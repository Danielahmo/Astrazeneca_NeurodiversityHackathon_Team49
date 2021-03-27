close all
clear all
% Video, in the real case it would be the live transmission from 
% the webcam
vid=VideoReader('video2.mp4');
vid.CurrentTime=0.0
% Reference image for processing, in the real case it would be an initial 
% calibration
ref = imread('referencia.jpg');
FDetector=vision.CascadeObjectDetector;
c=1;
while hasFrame(vid)
    %Detection of the frame for the eyes
    I = readFrame(vid);
    BB=step(FDetector,I);
    EyeDetector=vision.CascadeObjectDetector('EyePairBig');
    BB=step(EyeDetector,I);
    
    J = imhistmatch(I(:,:,1),ref(:,:,1));
    %Cut the image for processing the eyes
    fram = imcrop(J,BB(1,:));
    
    %Increase the contrast of the image
    fram = fram*7;
    %Image binarization
    bw = imcomplement(im2bw(imbinarize(fram)));
    %imshow(bw)
    
    %Select the 2 biggest objects
    bw = bwareafilt(bw,2);

    %Obtain circularity and centroid of the objects
    stats=table2array(regionprops('table',bw,'Circularity','Centroid'));
    if isempty(stats) == 1
        continue
    end
    % Selects the object with the major circularity
    eye1 = stats(stats(:,3)==max(stats(:,3)),1:2);
  
    % Delets the row with the object with more circularity
    stats2 = stats(stats(:,3)~= max(stats(:,3)),:);
    if isempty(stats2) == 1
        continue
    end
    %Selects the other eye with major circularity
    eye2 = stats2(stats2(:,3)==max(stats2(:,3)),1:2);
    
    %Inserts circle in the centroid
    
    im = insertShape(I,'circle',[(eye1(1)+BB(1,1)) (eye1(2)+BB(1,2))  2],'LineWidth',5);
    im = insertShape(im,'circle',[(eye2(1)+BB(1,1)) (eye2(2)+BB(1,2)) 2],'LineWidth',5);
    imshow(im)
    
    %Save coordinates from the eyes
    cord1(c,:)=eye1(:);
    cord2(c,:)=eye2(:);
    c=c+1;
 
end

%Standard deviation from the coordinates in the x axes
std1 = std(cord1(:,1));
std2 = std(cord2(:,2));
