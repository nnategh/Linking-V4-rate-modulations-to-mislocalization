%% plot modulation index

load("modulation_index.mat")
% load("modulation_index_selected_probes.mat")

t_win = 60:120;

figure
subplot(3,1,1)
line1 = niceplot3(1:300,[mod_idx_high1;mod_idx_high2],15,42/255,154/255,99/255);
hold on
yline(0,'k--','LineWidth',2)
line2 = niceplot3(1:300,[mod_idx_low1;mod_idx_low2],15,154/255,42/255,99/255);
xlabel('time from stimulus onset (ms)')
ylabel('modulation index (unit)')
xlim([0 200])
ylim([-0.4 0.4])
legend([line1 line2],'High','Low')

subplot(3,1,2)
high1 = mean(mod_idx_high1(:,t_win),2,'omitnan');
low1 = mean(mod_idx_low1(:,t_win),2,'omitnan');
high2 = mean(mod_idx_high2(:,t_win),2,'omitnan');
low2 = mean(mod_idx_low2(:,t_win),2,'omitnan');

p1 = signrank(high1, low1);
p2 = signrank(high2, low2);

scatter(high1, low1, 'filled')
hold on
scatter(high2, low2, 'filled')
axis square
axis([-1 1 -1 1])
plot([-1 1], [-1 1], '--k')
title(['n = ' num2str(size(high1,1)+size(high2,1))])
xlabel('modulation index (unit) [high]')
ylabel('modulation index (unit) [low]')

med_iqr_high1 = [nanmedian(high1) iqr(high1)];
med_iqr_low1 = [nanmedian(low1) iqr(low1)];
med_iqr_high2 = [nanmedian(high2) iqr(high2)];
med_iqr_low2 = [nanmedian(low2) iqr(low2)];

effect1 = meanEffectSize(high1,low1);
effect2 = meanEffectSize(high2,low2);

p = signrank([high1;high2], [low1;low2]);
subplot(3,1,3)
histogram([high1;high2] - [low1;low2],'BinWidth',0.1);
xlabel('difference (unit)')
ylabel('number of neurons')
title(['p = ' num2str(p)])