import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    author: z.string().default('Taewoon Kim'),
    tags: z.array(z.string()).optional().default([]),
    'cover-img': z.string().optional(),
    'thumbnail-img': z.string().optional(),
    mathjax: z.boolean().optional(),
  }),
});

export const collections = { blog };