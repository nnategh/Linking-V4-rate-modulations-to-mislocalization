%% plot center of mass

load("CoM_horizontal.mat")
% load("CoM_vertical.mat")

%% center of mass

figure(1)

subplot(2,2,1)
scatter(CoM_high_fix1,CoM_high_peri1,'filled')
hold on
scatter(CoM_low_fix1,CoM_low_peri1,'filled')
plot([0 12], [0 12],'k--')
axis([0 12 0 12])
axis square
xlabel('center of mass (dva) [fixation]')
ylabel('center of mass (dva) [perisaccadic]')
p_high1 = signrank(CoM_high_fix1,CoM_high_peri1);
p_low1 = signrank(CoM_low_fix1,CoM_low_peri1);
legend(['High: p = ' num2str(p_high1)], ['Low: p = ' num2str(p_low1)])
title('Monkey 1')

subplot(2,2,2)
histogram(CoM_high_fix1-CoM_high_peri1,'BinWidth',1)
hold on
histogram(CoM_low_fix1-CoM_low_peri1,'BinWidth',1)
xlabel('difference (dva)')
ylabel('number of neurons')

med_iqr_high_fix1 = [nanmedian(CoM_high_fix1) iqr(CoM_high_fix1)];
med_iqr_high_peri1 = [nanmedian(CoM_high_peri1) iqr(CoM_high_peri1)];
med_iqr_low_fix1 = [nanmedian(CoM_low_fix1) iqr(CoM_low_fix1)];
med_iqr_low_peri1 = [nanmedian(CoM_low_peri1) iqr(CoM_low_peri1)];

effect_high1 = meanEffectSize(CoM_high_fix1,CoM_high_peri1);
effect_low1 = meanEffectSize(CoM_low_fix1,CoM_low_peri1);

subplot(2,2,3)
scatter(CoM_high_fix2,CoM_high_peri2,'filled')
hold on
scatter(CoM_low_fix2,CoM_low_peri2,'filled')
plot([0 12], [0 12],'k--')
axis([0 12 0 12])
axis square
xlabel('center of mass (dva) [fixation]')
ylabel('center of mass (dva) [perisaccadic]')
p_high2 = signrank(CoM_high_fix2,CoM_high_peri2);
p_low2 = signrank(CoM_low_fix2,CoM_low_peri2);
legend(['High: p = ' num2str(p_high2)], ['Low: p = ' num2str(p_low2)])
title('Monkey 2')

subplot(2,2,4)
histogram(CoM_high_fix2-CoM_high_peri2,'BinWidth',1)
hold on
histogram(CoM_low_fix2-CoM_low_peri2,'BinWidth',1)
xlabel('difference (dva)')
ylabel('number of neurons')

med_iqr_high_fix2 = [nanmedian(CoM_high_fix2) iqr(CoM_high_fix2)];
med_iqr_high_peri2 = [nanmedian(CoM_high_peri2) iqr(CoM_high_peri2)];
med_iqr_low_fix2 = [nanmedian(CoM_low_fix2) iqr(CoM_low_fix2)];
med_iqr_low_peri2 = [nanmedian(CoM_low_peri2) iqr(CoM_low_peri2)];

effect_high2 = meanEffectSize(CoM_high_fix2,CoM_high_peri2);
effect_low2 = meanEffectSize(CoM_low_fix2,CoM_low_peri2);

%% center of mass shift

figure(2)

CoM_shift_high1 = CoM_high_peri1 - CoM_high_fix1;
CoM_shift_low1 = CoM_low_peri1 - CoM_low_fix1;
CoM_shift_high2 = CoM_high_peri2 - CoM_high_fix2;
CoM_shift_low2 = CoM_low_peri2 - CoM_low_fix2;

subplot(1,2,1)
scatter(CoM_shift_high1,CoM_shift_low1,'filled')
hold on
scatter(CoM_shift_high2,CoM_shift_low2,'filled')
plot([-6 6], [-6 6],'k--')
axis([-6 6 -6 6])
axis square
xlabel('center of mass shift (dva) [High]')
ylabel('center of mass shift (dva) [Low]')
legend('Monkey 1', 'Monkey 2')
title(['p = ' num2str(signrank([CoM_shift_high1;CoM_shift_high2],[CoM_shift_low1;CoM_shift_low2]))])

subplot(1,2,2)
histogram([CoM_shift_high1;CoM_shift_high2]-[CoM_shift_low1;CoM_shift_low2],'BinWidth',1)
hold on
xlabel('difference (dva)')
ylabel('number of neurons')
title(['n = ' num2str(size([CoM_shift_high1;CoM_shift_high2],1))])

p1 = signrank(CoM_shift_high1,CoM_shift_low1);
p2 = signrank(CoM_shift_high2,CoM_shift_low2);

med_iqr_high1 = [nanmedian(CoM_shift_high1) iqr(CoM_shift_high1)];
med_iqr_low1 = [nanmedian(CoM_shift_low1) iqr(CoM_shift_low1)];
med_iqr_high2 = [nanmedian(CoM_shift_high2) iqr(CoM_shift_high2)];
med_iqr_low2 = [nanmedian(CoM_shift_low2) iqr(CoM_shift_low2)];

effect1 = meanEffectSize(CoM_shift_high1,CoM_shift_low1);
effect2 = meanEffectSize(CoM_shift_high2,CoM_shift_low2);