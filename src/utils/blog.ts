import type { CollectionEntry } from 'astro:content';

export type BlogEntry = CollectionEntry<'blog'>;
export type BlogTagGroup = {
  name: string;
  slug: string;
  aliases: string[];
  posts: BlogEntry[];
  count: number;
};

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

export function slugifyTag(tag: string) {
  return tag
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function getTagUrl(tag: string) {
  return `/tags/${slugifyTag(tag)}/`;
}

function getPreferredTagName(tags: string[]) {
  return [...tags].sort((left, right) => {
    const score = (value: string) => {
      let total = 0;

      if (/^[A-Z0-9]+$/.test(value)) total += 4;
      if (/[A-Z]{2,}/.test(value)) total += 2;
      if (value.includes(' ')) total += 2;
      if (!value.includes('-')) total += 1;

      return total;
    };

    return score(right) - score(left) || left.localeCompare(right);
  })[0];
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

export function getAllTagGroups(posts: BlogEntry[]): BlogTagGroup[] {
  const groups = new Map<
    string,
    {
      aliases: Set<string>;
      posts: BlogEntry[];
      postIds: Set<string>;
    }
  >();

  for (const post of posts) {
    for (const tag of post.data.tags ?? []) {
      const slug = slugifyTag(tag);
      const group = groups.get(slug) ?? {
        aliases: new Set<string>(),
        posts: [],
        postIds: new Set<string>(),
      };

      group.aliases.add(tag);

      if (!group.postIds.has(post.id)) {
        group.postIds.add(post.id);
        group.posts.push(post);
      }

      groups.set(slug, group);
    }
  }

  return [...groups.entries()]
    .map(([slug, group]) => {
      const aliases = [...group.aliases].sort((left, right) => left.localeCompare(right));

      return {
        name: getPreferredTagName(aliases),
        slug,
        aliases,
        posts: group.posts,
        count: group.posts.length,
      };
    })
    .sort((left, right) => left.name.localeCompare(right.name));
}