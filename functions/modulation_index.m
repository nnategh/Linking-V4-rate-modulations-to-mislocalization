function mod_in = modulation_index(group1, group2)

bin = 15;
mod_in = nan(length(group1),300);
for n = 1:length(group1)
    if ~isempty(group1{n,:}) && ~isempty(group2{n,:})
        for d = 1:size(group1{n,:},2)-bin       
            data1 = mean(mean(group1{n,:}(:,d:d+bin),2,'omitnan'),'omitnan');
            data2 = mean(mean(group2{n,:}(:,d:d+bin),2,'omitnan'),'omitnan');
            if data1 + data2 ~= 0
                mod_in(n,d) = (data2 - data1) / (data2 + data1);
            else
                mod_in(n,d) = nan;
            end
        end
    end
    n
end

end