%% plot mislocalization and localization error

load("misloc_locerr.mat") 
%load("misloc_locerr_selected_probes.mat")

%% mislocalization

time = -250:100;

figure(1)

subplot(2,1,1)
niceplot3(time,mislocalization1,20,115/255,61/255,44/255);
ylabel('mislocalization (dva)')
title('Monkey 1')

subplot(2,1,2)
niceplot3(time,mislocalization2,20,115/255,61/255,44/255);
xlabel('time of stimulus presentation from saccade onset (ms)')
ylabel('mislocalization (dva)')
title('Monkey 2')


%% localization error

figure(2)

subplot(2,2,1)
scatter(locerr_fix1,locerr_peri1,'filled') 
xlabel('localization error [fixation]')
ylabel('localization error [perisaccadic]')
axis square
axis([-4 4 -4 4])
title(['Monkey 1: n = ' num2str(size(locerr_fix1,1)) ', p = ' num2str(signrank(locerr_fix1,locerr_peri1))])
subplot(2,2,2)
histogram((locerr_fix1-locerr_peri1))
xlabel('difference (dva)')
ylabel('number of probes')

med_iqr_fix1 = [median(locerr_fix1) iqr(locerr_fix1)];
med_iqr_peri1 = [median(locerr_peri1) iqr(locerr_peri1)];
effect1 = meanEffectSize(locerr_fix1,locerr_peri1);

subplot(2,2,3)
scatter(locerr_fix2,locerr_peri2,'filled') 
xlabel('localization error [fixation]')
ylabel('localization error [perisaccadic]')
axis square
axis([-4 4 -4 4])
title(['Monkey 2: n = ' num2str(size(locerr_fix2,1)) ', p = ' num2str(signrank(locerr_fix2,locerr_peri2))])
subplot(2,2,4)
histogram(locerr_fix2-locerr_peri2)
xlabel('difference (dva)')
ylabel('number of probes')

med_iqr_fix2 = [median(locerr_fix2) iqr(locerr_fix2)];
med_iqr_peri2 = [median(locerr_peri2) iqr(locerr_peri2)];
effect2 = meanEffectSize(locerr_fix2,locerr_peri2);