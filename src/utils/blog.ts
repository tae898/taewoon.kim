import type { CollectionEntry } from 'astro:content';

export type BlogEntry = CollectionEntry<'blog'>;

const dateFormatter = new Intl.DateTimeFormat('en', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

export function parsePostId(id: string) {
  const match = id.match(/^(\d{4})-(\d{2})-(\d{2})-(.+?)(?:\.md)?$/);

  if (!match) {
    throw new Error(`Unexpected blog post id: ${id}`);
  }

  const [, year, month, day, slug] = match;

  return {
    year,
    month,
    day,
    slug,
    date: new Date(`${year}-${month}-${day}T00:00:00Z`),
  };
}

export function getPostUrl(id: string) {
  const { year, month, day, slug } = parsePostId(id);
  return `/${year}/${month}/${day}/${slug}/`;
}

export function sortPosts(posts: BlogEntry[]) {
  return [...posts].sort(
    (left, right) => parsePostId(right.id).date.getTime() - parsePostId(left.id).date.getTime(),
  );
}

export function formatPostDate(id: string) {
  return dateFormatter.format(parsePostId(id).date);
}

export function getPostImage(post: BlogEntry) {
  return post.data['thumbnail-img'] ?? post.data['cover-img'];
}

export function createExcerpt(source: string, length = 180) {
  const text = source
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/[#>*_~\-]+/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$]*\$/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (text.length <= length) {
    return text;
  }

  return `${text.slice(0, length).trimEnd()}...`;
}

export function getAllTags(posts: BlogEntry[]) {
  return [...new Set(posts.flatMap((post) => post.data.tags ?? []))].sort((left, right) =>
    left.localeCompare(right),
  );
}