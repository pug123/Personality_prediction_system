# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:26:39 2018

@author: Shawlock
"""

ps = pd.read_csv('ForumMessages.csv')

ps['Message'] = ps['Message'].fillna('')
ps_join = ps.groupby('AuthorUserId')['Message'].agg(lambda col: ' '.join(col)).reset_index()

ps_join['clean_comments'] = ps_join['Message'].apply(cleanText)

personality['clean_posts'] = personality['posts'].apply(cleanText)

model_lr.fit(personality['clean_posts'], personality['type'])
pred_all = model_lr.predict(ps_join['clean_comments'])

cnt_all = np.unique(pred_all, return_counts=True)

pred_df = pd.DataFrame({'personality': cnt_all[0], 'count': cnt_all[1]},
                      columns=['personality', 'count'], index=None)

pred_df.sort_values('count', ascending=False, inplace=True)

plt.figure(figsize=(12,6))
sns.barplot(pred_df['personality'], pred_df['count'], alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Personality', fontsize=12)
plt.show()


pred_df['percent'] = pred_df['count']/pred_df['count'].sum()

pred_df['description'] = pred_df['personality'].apply(lambda x: ' '.join([mbti[l] for l in list(x)]))

pred_df