import { type NextRequest, NextResponse } from "next/server"

// This is a proxy endpoint to handle CORS issues when calling the external API
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // Log what we're sending to help debug
    console.log("Sending request to API with form data keys:", [...formData.keys()])

    // Create a new FormData object to ensure proper formatting
    const newFormData = new FormData()

    // Add each file to the new FormData with the expected key format
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`Adding file: ${key}, type: ${value.type}, size: ${value.size}`)
        newFormData.append("files", value) // The API might expect "file" as the key
      }
    }

    // Set up fetch options with appropriate headers
    const fetchOptions = {
      method: "POST",
      body: newFormData,
      headers: {
        // Don't set Content-Type header when sending FormData
        // It will be set automatically with the correct boundary
      },
    }

    console.log("Attempting to fetch from API...")

    // Forward the request to the actual API
    const response = await fetch("https://covid-api-1001214075164.us-central1.run.app/predict/", fetchOptions)

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`API responded with status: ${response.status}, body: ${errorText}`)
      throw new Error(`API responded with status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Proxy error details:", error)

    // Return a more detailed error message
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Unknown error",
        details: error instanceof Error ? error.stack : "No stack trace available",
      },
      { status: 500 },
    )
  }
}

