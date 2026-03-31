import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { site } from '../site';
import { createExcerpt, getPostUrl, parsePostId, sortPosts } from '../utils/blog';

export async function GET(context) {
  const posts = sortPosts(await getCollection('blog'));

  return rss({
    title: site.title,
    description: site.description,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.subtitle ?? createExcerpt(post.body),
      pubDate: parsePostId(post.id).date,
      link: getPostUrl(post.id),
      categories: post.data.tags ?? [],
      author: post.data.author,
    })),
  });
}