import { NextResponse } from 'next/server';
import { WikiStructure } from '@/types/wiki/wikistructure';
import { WikiPage } from '@/types/wiki/wikipage';
import { promises as fs } from 'fs';
import path from 'path';

/**
 * GET handler for retrieving wiki content by ID from cache
 * @param request - The incoming request
 * @param params - URL parameters containing the wiki ID
 * @returns The wiki content or an error response
 */
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const id = params.id;
    if (!id) {
      return NextResponse.json(
        { error: 'Wiki ID is required' },
        { status: 400 }
      );
    }

    // Get cached wiki data from .adalflow/databases directory
    const cacheDir = '.adalflow/databases';
    let cacheData;

    try {
      // List all cache files
      const files = await fs.readdir(cacheDir);
      
      // Find the latest cache file for the wiki
      const cacheFile = files
        .filter((f: string) => f.endsWith('.pkl'))
        .sort()
        .reverse()[0];

      if (!cacheFile) {
        return NextResponse.json(
          { error: 'No cached wiki data found' },
          { status: 404 }
        );
      }

      // Load the cached data
      const cachePath = path.join(cacheDir, cacheFile);
      const fileContent = await fs.readFile(cachePath, 'utf-8');
      cacheData = JSON.parse(fileContent);
    } catch (err) {
      console.error('Error accessing cache:', err);
      return NextResponse.json(
        { error: 'Error accessing wiki cache' },
        { status: 500 }
      );
    }

    if (!cacheData) {
      return NextResponse.json(
        { error: 'Failed to load cache data' },
        { status: 500 }
      );
    }

    // Find the wiki content by ID
    const wiki = cacheData.find((w: WikiStructure | WikiPage) => w.id === id);

    if (!wiki) {
      return NextResponse.json(
        { error: 'Wiki content not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(wiki);

  } catch (error: unknown) {
    console.error('Error retrieving wiki:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
