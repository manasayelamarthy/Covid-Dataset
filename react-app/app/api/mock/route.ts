import { type NextRequest, NextResponse } from "next/server"

// This is a mock API endpoint that returns data in the format provided
export async function POST(request: NextRequest) {
  try {
    // Parse the FormData to get the images
    const formData = await request.formData()

    // Count how many images were uploaded
    const imageCount = Array.from(formData.entries()).filter(([key, value]) => value instanceof File).length

    if (imageCount === 0) {
      return NextResponse.json({ error: "No images provided" }, { status: 400 })
    }

    // Simulate processing time
    await new Promise((resolve) => setTimeout(resolve, 1000))

    // Create mock predictions based on the number of images
    // Using the format from the user's example
    const labels = ["Viral Pneumonia", "Covid", "Normal"]
    const predictions = Array(imageCount)
      .fill(null)
      .map(() => {
        const label = labels[Math.floor(Math.random() * labels.length)]
        // Generate a confidence between 95 and 100
        const confidence = 95 + Math.random() * 5

        return {
          label,
          confidence: Number(confidence.toFixed(2)),
        }
      })

    // Return in the exact format provided by the user
    return NextResponse.json({
      predictions,
    })
  } catch (error) {
    console.error("Mock API error:", error)
    return NextResponse.json({ error: "Failed to process images" }, { status: 500 })
  }
}

