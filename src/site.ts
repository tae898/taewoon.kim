export type SiteLink = {
  label: string;
  href: string;
};

export const cvLink: SiteLink = {
  label: 'CV',
  href: '/taewoon-kim-cv-2026-04-22.pdf',
};

export const socialLinks: SiteLink[] = [
  { label: 'Email', href: 'mailto:taewoon@humem.ai' },
  { label: 'GitHub', href: 'https://github.com/tae898' },
  { label: 'LinkedIn', href: 'https://linkedin.com/in/tae898' },
  { label: 'X', href: 'https://twitter.com/tae898' },
  { label: 'YouTube', href: 'https://www.youtube.com/@tae898' },
  { label: 'Scholar', href: 'https://scholar.google.com/citations?user=dJ4ksGoAAAAJ&hl=en' },
];

export const homePage = {
  intro: {
    id: 'intro',
    title: "I'm an AI researcher and engineer.",
    lede: 'I care about building toward AGI through better memory, stronger systems, and practical engineering.',
  },
  blog: {
    id: 'blog',
    label: 'Blog',
    description: 'Browse recent posts.',
  },
} as const;

export const socialVariants = {
  about: {
    includeCv: true,
    excludeLabels: ['YouTube'],
  },
  footer: {
    includeCv: true,
    excludeLabels: ['YouTube'],
  },
  contact: {
    includeCv: true,
    excludeLabels: [],
  },
} as const;

export function getSocialLinks(variant: keyof typeof socialVariants): SiteLink[] {
  const config = socialVariants[variant];
  const excluded = new Set<string>(config.excludeLabels);
  const links = socialLinks.filter((item) => !excluded.has(item.label));

  if (config.includeCv && !excluded.has(cvLink.label)) {
    return [...links, cvLink];
  }

  return links;
}

export const site = {
  title: 'Taewoon Kim',
  description: 'AI researcher and engineer.',
  author: 'Taewoon Kim',
  url: 'https://taewoon.kim',
  gtag: 'G-8E02EYZXEW',
  avatar: '/assets/img/TAE.png',
  socialImage: '/assets/img/TAE.png',
  nav: [
    { label: 'About', href: `/#${homePage.intro.id}` },
    { label: homePage.blog.label, href: `/#${homePage.blog.id}` },
  ],
  cv: cvLink.href,
  social: socialLinks,
};