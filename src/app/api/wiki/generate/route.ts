import { NextRequest, NextResponse } from 'next/server';

/**
 * API route for generating a wiki report using the backend API
 * This route proxies requests to the backend API endpoint
 */
export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const requestBody = await request.json();

    // Forward the request to the backend API
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/wiki/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    // Check if the response is successful
    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate wiki report' },
        { status: response.status }
      );
    }

    // Return the response from the backend API
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error generating wiki report:', error);
    return NextResponse.json(
      { error: 'An unexpected error occurred while generating the wiki report' },
      { status: 500 }
    );
  }
}
