img = imread('lena.png');
img = rgb2gray(img);
%tamanho da janela(Kernel)
window = 3;
kernel = ones(window, window);

%imshow(img);

iNoiseUm = imnoise(img, 'salt & pepper',0.02);
iNoiseDois = imnoise(img, 'salt & pepper',0.05);
%imshow([iNoiseUm,iNoiseDois]);

%tenho que percorrer a imagem 8 em * pixels na
%horizontal e na vertical
for i = 1: window: 512
    %percorre na vertical
    disp(i);
    %kernel * img(i i+8,
    for j = 1:window: 512
        %tenho q calcular a media
        if(((i+window)< 512) && ((j+window)<512))
            part = iNoiseUm(i : i+(window-1), j : j+(window-1));
            media = cancMediaJanela(part,window);
            mediana = median(part(:));
            iNoiseUm(i : i+(window-1), j : j+(window-1))  = media;
            iNoiseUm(i : i+(window-1), j : j+(window-1))  = mediana;
        
        end
    %percorre na horizonal
    end
end
%disp(media);
imshow(iNoiseUm);

function y = cancMediaJanela(window,sizeWindow)
sum = int64(0);
for i=1:sizeWindow

    for j = 1 :sizeWindow
        sum = sum +  int64(window(i,j));
        
    end

end
y = uint8(sum/(sizeWindow^2));

end




%part of image
%imshow(img(1:100 , 1:100));
