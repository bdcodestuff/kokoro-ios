#if canImport(eSpeakNGLib)

import Foundation
import eSpeakNGLib
import MisakiSwift

class eSpeakNGG2PProcessor : G2PProcessor {
  private var eSpeakEngine: eSpeakNG?

  func setLanguage(_ language: Language) throws {
    eSpeakEngine = try eSpeakNG()
    
    if let language = eSpeakNG.Language(rawValue: language.rawValue), let eSpeakEngine {
      try eSpeakEngine.setLanguage(language: language)
    } else {
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  func process(input: String) throws -> String {
    guard let eSpeakEngine else { throw G2PProcessorError.processorNotInitialized }
    return try eSpeakEngine.phonemize(text: input)
  }

  #if canImport(MisakiSwift)
  func processWithTokens(input: String) throws -> (phonemes: String, tokens: [MisakiSwift.MToken]) {
    // eSpeakNG doesn't support token-level information, so return empty tokens array
    let phonemes = try process(input: input)
    return (phonemes: phonemes, tokens: [])
  }
  #endif
}

#endif
